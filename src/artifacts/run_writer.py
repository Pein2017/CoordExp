"""Concrete single-writer artifacts for one CoordExp training run."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts.checkpoint_payload import (
    admit_inference_checkpoint_payload_identity,
)
from src.artifacts.resources import (
    merge_resource_high_water,
    validate_rank_cpu_resource_receipt,
)
from src.common.errors import ArtifactContractError


_MAX_WARNING_CODES = 32
_MAX_WARNING_COUNT = 2**31 - 1
_MAX_CHECKPOINT_PUBLICATION_EVENTS = 100_000
_OTHER_WARNING_CODE = "other"
_UNPHASED_FAILURE_PHASE = "unphased_failure"
_CONTINUATION_LINEAGE_FIELDS = frozenset(
    {
        "continuation_index",
        "parent_checkpoint_identity",
        "parent_continuation_index",
        "parent_run_id",
        "parent_segment_id",
    }
)
_PARENT_CHECKPOINT_IDENTITY_FIELDS = frozenset(
    {
        "checkpoint_step",
        "resolved_path",
        "training_state_aggregate_digest",
        "training_state_manifest_file_sha256",
    }
)
_COMMITTED_PROGRESS_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "completed_steps",
        "consumed_packs",
        "optimizer_update_status",
        "finite_status",
    }
)
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
_POLICY_IDENTITY_NAMES = frozenset(
    {
        "attention_proof",
        "cache",
        "eval_reduction",
        "input_provider",
        "packing",
        "profile_sync_timings",
        "resume",
        "runtime_determinism",
        "upstream_runtime_baseline",
    }
)
_PHASE_NAMES = frozenset(
    {
        "cache_admission",
        "cache_identity_resolution",
        "cache_preparation",
        "cache_publication",
        "cache_publication_admission",
        "checkpoint_publication",
        "config_provenance_resolution",
        "evaluation_hydration",
        "evaluation_execution",
        "first_optimizer_step",
        "model_loading",
        "optimizer_runtime_assembly",
        "steady_state",
        "train_rank_hydration",
        _UNPHASED_FAILURE_PHASE,
    }
)


@dataclass(frozen=True)
class RunWriter:
    """Own the fixed, rank-zero-written files in a training run directory."""

    run_dir: Path

    @classmethod
    def initialize(
        cls,
        *,
        run_dir: Path,
        run_id: str,
        run_name: str,
        artifact_root: Path,
        collision_outcome: str,
        created_at: str,
        config_fingerprint: str,
        resolved_config: Mapping[str, Any],
        world_size: int,
        resolved_max_steps: int | None = None,
        provenance: Mapping[str, Any] | None = None,
        measurement_context: Mapping[str, Any] | None = None,
        entry_started_at: str | None = None,
        segment_id: str | None = None,
        continuation_lineage: Mapping[str, Any] | None = None,
    ) -> "RunWriter":
        if world_size <= 0 or (
            resolved_max_steps is not None and resolved_max_steps <= 0
        ):
            raise ArtifactContractError(
                "world size and resolved maximum steps must be positive",
                code="run_writer.invalid_runtime_summary",
            )
        resolved_segment_id = (
            uuid.uuid4().hex if segment_id is None else _validate_segment_id(segment_id)
        )
        continuation_payload = _continuation_payload(
            run_id=run_id,
            segment_id=resolved_segment_id,
            lineage=continuation_lineage,
        )
        run_dir = run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ArtifactContractError(
                "run directory must be empty",
                code="run_writer.run_dir_not_empty",
                context={"run_dir": str(run_dir)},
            )
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = run_dir.parent / f".{run_dir.name}.{uuid.uuid4().hex}.init"
        staging_dir.mkdir()
        staging_writer = cls(run_dir=staging_dir)
        provenance_payload = _provenance_payload(provenance)
        measurement_payload = _measurement_payload(
            measurement_context,
            entry_started_at=created_at
            if entry_started_at is None
            else entry_started_at,
        )
        try:
            staging_writer._write_json_atomic(
                staging_writer.resolved_config_path, dict(resolved_config)
            )
            staging_writer._write_json_atomic(
                staging_writer.run_path,
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "artifact_root": str(artifact_root.resolve()),
                    "collision_outcome": collision_outcome,
                    "status": "initialized",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "completed_at": None,
                    "resolved_config_path": "resolved_config.json",
                    "config_fingerprint": config_fingerprint,
                    "runtime": {"world_size": world_size},
                    "resolved_max_steps": resolved_max_steps,
                    "completed_steps": 0,
                    "consumed_packs": 0,
                    "checkpoint_event_count": 0,
                    "final_optimizer_update_status": None,
                    "final_finite_status": None,
                    "terminal_error": None,
                    "warning_counts": {},
                    "materializations": {},
                    "forward_input_provider_mode": None,
                    "forward_input_provider_resolution": None,
                    "provenance": provenance_payload,
                    "policy_identities": {},
                    "continuation": continuation_payload,
                    "measurement": measurement_payload,
                },
            )
            staging_writer.logging_path.touch(exist_ok=False)
            _fsync_directory(staging_dir)
            os.replace(staging_dir, run_dir)
            _fsync_directory(run_dir.parent)
        except BaseException:
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        return cls(run_dir=run_dir)

    @property
    def run_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "resolved_config.json"

    @property
    def logging_path(self) -> Path:
        return self.run_dir / "logging.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def read_run(self) -> dict[str, Any]:
        return json.loads(self.run_path.read_text(encoding="utf-8"))

    def append_logging_row(self, row: Mapping[str, Any]) -> Path:
        normalized = _normalize_logging_row(row)
        encoded = _strict_json_dumps(normalized, compact=True)
        with self.logging_path.open("a", encoding="utf-8") as handle:
            # Serialize completely before opening and issue one append write per row.
            handle.write(encoded + "\n")
            handle.flush()
        return self.logging_path

    def record_warning(self, code: str, *, count: int = 1) -> None:
        if not code or count <= 0:
            raise ArtifactContractError(
                "warning code must be nonempty and count must be positive",
                code="run_writer.invalid_warning",
            )
        state = self.read_run()
        counts = state["warning_counts"]
        target = code
        reserved_other_slot = _OTHER_WARNING_CODE not in counts
        if (
            code not in counts
            and len(counts) >= _MAX_WARNING_CODES - reserved_other_slot
        ):
            target = _OTHER_WARNING_CODE
        counts[target] = min(_MAX_WARNING_COUNT, counts.get(target, 0) + count)
        self._write_json_atomic(self.run_path, state)

    def bind_materialization(
        self,
        split: str,
        *,
        cache_format_version: int | str,
        semantic_fingerprint: str,
        determinant_digest: str,
    ) -> None:
        if split not in {"train", "eval"}:
            raise ArtifactContractError(
                "materialization split must be train or eval",
                code="run_writer.invalid_materialization_split",
            )
        state = self.read_run()
        if split in state["materializations"]:
            raise ArtifactContractError(
                "materialization binding is immutable",
                code="run_writer.materialization_already_bound",
                context={"split": split},
            )
        binding = {
            "cache_format_version": cache_format_version,
            "semantic_fingerprint": semantic_fingerprint,
            "determinant_digest": determinant_digest,
        }
        state["materializations"][split] = binding
        self._write_json_atomic(self.run_path, state)

    def bind_schedule(self, *, resolved_max_steps: int) -> None:
        """Bind the epoch-derived training plan exactly once after initialization."""
        if resolved_max_steps <= 0:
            raise ArtifactContractError(
                "resolved maximum steps must be positive",
                code="run_writer.invalid_resolved_max_steps",
            )
        state = self.read_run()
        current = state.get("resolved_max_steps")
        if current is not None:
            raise ArtifactContractError(
                "resolved training schedule is immutable",
                code="run_writer.schedule_already_bound",
            )
        state["resolved_max_steps"] = resolved_max_steps
        self._write_json_atomic(self.run_path, state)

    def bind_forward_input_provider_mode(
        self,
        mode: str,
        *,
        resolution: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the resolved forward-input provider mode exactly once."""
        if mode not in {"legacy_fused", "overlapped", "synchronous"}:
            raise ArtifactContractError(
                "forward input provider mode must be 'legacy_fused', 'overlapped', or 'synchronous'",
                code="run_writer.invalid_forward_input_provider_mode",
                context={"mode": mode},
            )
        state = self.read_run()
        current = state.get("forward_input_provider_mode")
        if (
            current is not None
            or state.get("forward_input_provider_resolution") is not None
        ):
            raise ArtifactContractError(
                "forward input provider mode is immutable",
                code="run_writer.forward_input_provider_mode_already_bound",
            )
        resolution_payload = (
            None
            if resolution is None
            else _strict_mapping(resolution, field="forward input provider resolution")
        )
        if (
            resolution_payload is not None
            and resolution_payload.get("resolved_mode") != mode
        ):
            raise ArtifactContractError(
                "forward input provider resolution disagrees with its bound mode",
                code="run_writer.forward_input_provider_resolution_mismatch",
                context={"mode": mode},
            )
        state["forward_input_provider_mode"] = mode
        state["forward_input_provider_resolution"] = resolution_payload
        self._write_json_atomic(self.run_path, state)

    def bind_policy_identity(self, name: str, identity: Mapping[str, Any]) -> None:
        """Bind one additive resolved policy identity exactly once."""

        if name not in _POLICY_IDENTITY_NAMES:
            raise ArtifactContractError(
                "policy identity name is not supported",
                code="run_writer.invalid_policy_identity_name",
                context={"name": name},
            )
        payload = _strict_mapping(identity, field="policy identity")
        state = self.read_run()
        policies = state["policy_identities"]
        if name in policies:
            raise ArtifactContractError(
                "policy identity binding is immutable",
                code="run_writer.policy_identity_already_bound",
                context={"name": name},
            )
        policies[name] = payload
        self._write_json_atomic(self.run_path, state)

    def bind_continuation_lineage(self, lineage: Mapping[str, Any]) -> None:
        """Bind an admitted parent exactly once before any training progress."""

        state = self.read_run()
        current = state["continuation"]
        if current.get("parent") is not None:
            raise ArtifactContractError(
                "continuation lineage is already bound",
                code="run_writer.continuation_lineage_already_bound",
            )
        if (
            state.get("status") != "initialized"
            or state.get("completed_steps") != 0
            or state.get("consumed_packs") != 0
            or state.get("checkpoint_event_count") != 0
        ):
            raise ArtifactContractError(
                "continuation lineage must be bound before training progress",
                code="run_writer.continuation_lineage_too_late",
            )
        state["continuation"] = _continuation_payload(
            run_id=state["run_id"],
            segment_id=current["segment_id"],
            lineage=lineage,
        )
        self._write_json_atomic(self.run_path, state)

    def begin_phase(
        self,
        phase: str,
        *,
        started_at: str,
        resources: Mapping[str, Any] | None = None,
    ) -> None:
        """Start one named, non-overlapping training phase."""

        _validate_phase_name(phase)
        _validate_timestamp(started_at, field="started_at")
        state = self.read_run()
        measurement = state["measurement"]
        active = measurement["active_phase"]
        if active is not None:
            raise ArtifactContractError(
                "a run phase is already active",
                code="run_writer.phase_already_active",
                context={"active_phase": active, "requested_phase": phase},
            )
        if phase in measurement["phases"]:
            raise ArtifactContractError(
                "a completed or failed phase cannot be reopened",
                code="run_writer.phase_already_recorded",
                context={"phase": phase},
            )
        measurement["active_phase"] = phase
        measurement["terminal_phase"] = phase
        measurement["terminal_phase_status"] = "running"
        measurement["phase_order"].append(phase)
        measurement["phases"][phase] = {
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "duration_seconds": None,
        }
        _merge_resources(measurement, resources)
        self._write_json_atomic(self.run_path, state)

    def record_phase_not_run(self, phase: str, *, reason: str) -> None:
        """Record an explicitly inapplicable phase without claiming execution."""

        _validate_phase_name(phase)
        if not isinstance(reason, str) or not reason:
            raise ArtifactContractError(
                "not-run phase reason must be nonempty",
                code="run_writer.invalid_phase_status",
            )
        state = self.read_run()
        measurement = state["measurement"]
        if measurement["active_phase"] is not None:
            raise ArtifactContractError(
                "a not-run phase cannot be recorded while another phase is active",
                code="run_writer.phase_already_active",
                context={"active_phase": measurement["active_phase"]},
            )
        if phase in measurement["phases"]:
            raise ArtifactContractError(
                "a run phase can be recorded only once",
                code="run_writer.phase_already_recorded",
                context={"phase": phase},
            )
        measurement["phases"][phase] = {
            "status": "not_run",
            "reason": reason,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0.0,
        }
        measurement["phase_order"].append(phase)
        self._write_json_atomic(self.run_path, state)

    def record_completed_phase(
        self,
        phase: str,
        *,
        completed_at: str,
        duration_seconds: float,
        resources: Mapping[str, Any] | None = None,
        rank_resources: Mapping[str, Any] | None = None,
        rank_details: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a completed nested/pre-writer phase from its measured duration."""

        _validate_phase_name(phase)
        _validate_timestamp(completed_at, field="completed_at")
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, (int, float))
            or not math.isfinite(float(duration_seconds))
            or float(duration_seconds) < 0.0
        ):
            raise ArtifactContractError(
                "phase duration must be finite and nonnegative",
                code="run_writer.invalid_phase_duration",
            )
        state = self.read_run()
        measurement = state["measurement"]
        if measurement["active_phase"] is not None:
            raise ArtifactContractError(
                "a completed summary cannot be recorded while another phase is active",
                code="run_writer.phase_already_active",
                context={"active_phase": measurement["active_phase"]},
            )
        if phase in measurement["phases"]:
            raise ArtifactContractError(
                "a run phase can be recorded only once",
                code="run_writer.phase_already_recorded",
                context={"phase": phase},
            )
        receipt = {
            "status": "completed",
            "started_at": None,
            "completed_at": completed_at,
            "duration_seconds": float(duration_seconds),
        }
        _attach_rank_phase_receipt(
            receipt,
            rank_resources=rank_resources,
            rank_details=rank_details,
        )
        measurement["phases"][phase] = receipt
        measurement["phase_order"].append(phase)
        measurement["terminal_phase"] = phase
        measurement["terminal_phase_status"] = "completed"
        measurement["last_completed_phase"] = phase
        _merge_resources(measurement, resources)
        self._write_json_atomic(self.run_path, state)

    def record_failed_phase(
        self,
        phase: str,
        *,
        completed_at: str,
        duration_seconds: float,
        reason: str,
        rank_resources: Mapping[str, Any] | None = None,
        rank_details: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one failed pre-writer or converged phase without orphaning it."""

        _validate_phase_name(phase)
        _validate_timestamp(completed_at, field="completed_at")
        _validate_nonnegative_finite(duration_seconds, field="duration_seconds")
        if not isinstance(reason, str) or not reason or len(reason) > 256:
            raise ArtifactContractError(
                "failed phase reason must be nonempty and bounded",
                code="run_writer.invalid_phase_status",
            )
        state = self.read_run()
        measurement = state["measurement"]
        if measurement["active_phase"] is not None:
            raise ArtifactContractError(
                "a failed summary cannot be recorded while another phase is active",
                code="run_writer.phase_already_active",
                context={"active_phase": measurement["active_phase"]},
            )
        if phase in measurement["phases"]:
            raise ArtifactContractError(
                "a run phase can be recorded only once",
                code="run_writer.phase_already_recorded",
                context={"phase": phase},
            )
        receipt: dict[str, Any] = {
            "status": "failed",
            "reason": reason,
            "started_at": None,
            "completed_at": completed_at,
            "duration_seconds": float(duration_seconds),
        }
        _attach_rank_phase_receipt(
            receipt,
            rank_resources=rank_resources,
            rank_details=rank_details,
        )
        measurement["phases"][phase] = receipt
        measurement["phase_order"].append(phase)
        measurement["terminal_phase"] = phase
        measurement["terminal_phase_status"] = "failed"
        measurement["failure_phase"] = phase
        measurement["steady_state_eligible"] = False
        self._write_json_atomic(self.run_path, state)

    def record_phase_summary(
        self,
        phase: str,
        *,
        status: str,
        completed_at: str | None,
        duration_seconds: float,
        duration_scope: str,
        accepted_measured_steps: int | None = None,
        expected_measured_steps: int | None = None,
        event_count: int | None = None,
        reason: str | None = None,
        resource_observation_scope: str | None = None,
        resource_high_water_observed_after_events: Mapping[str, Any] | None = None,
        rank_resources: Mapping[str, Any] | None = None,
        rank_details: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically persist one aggregate phase receipt without opening a phase."""

        if phase not in {"steady_state", "evaluation_execution"}:
            raise ArtifactContractError(
                "phase summary is supported only for aggregate execution phases",
                code="run_writer.invalid_phase_summary_name",
                context={"phase": phase},
            )
        if status not in {"completed", "failed", "not_run"}:
            raise ArtifactContractError(
                "phase summary status must be completed, failed, or not_run",
                code="run_writer.invalid_phase_status",
                context={"status": status},
            )
        if status == "not_run":
            if completed_at is not None or not isinstance(reason, str) or not reason:
                raise ArtifactContractError(
                    "a not-run phase summary requires a reason and no completion time",
                    code="run_writer.invalid_phase_summary",
                )
        else:
            _validate_timestamp(completed_at, field="completed_at")
        _validate_nonnegative_finite(duration_seconds, field="duration_seconds")
        if not isinstance(duration_scope, str) or not duration_scope:
            raise ArtifactContractError(
                "phase summary duration scope must be nonempty",
                code="run_writer.invalid_phase_summary",
            )

        receipt: dict[str, Any] = {
            "status": status,
            "started_at": None,
            "completed_at": completed_at,
            "duration_seconds": float(duration_seconds),
            "duration_scope": duration_scope,
        }
        if reason is not None:
            receipt["reason"] = reason

        if phase == "steady_state":
            accepted = _validate_nonnegative_int(
                accepted_measured_steps,
                field="accepted_measured_steps",
            )
            expected = _validate_nonnegative_int(
                expected_measured_steps,
                field="expected_measured_steps",
            )
            if event_count is not None:
                raise ArtifactContractError(
                    "event count does not belong to steady-state summary",
                    code="run_writer.invalid_phase_summary",
                )
            receipt.update(
                accepted_measured_steps=accepted,
                expected_measured_steps=expected,
                acceptance_scope=(
                    "applied_finite_post_warmup_steps_with_finite_"
                    "all_rank_max_step_duration"
                ),
            )
        else:
            events = _validate_nonnegative_int(event_count, field="event_count")
            if (
                accepted_measured_steps is not None
                or expected_measured_steps is not None
            ):
                raise ArtifactContractError(
                    "measured step counts do not belong to evaluation summary",
                    code="run_writer.invalid_phase_summary",
                )
            receipt["event_count"] = events
            if resource_observation_scope is not None:
                if (
                    not isinstance(resource_observation_scope, str)
                    or not resource_observation_scope
                ):
                    raise ArtifactContractError(
                        "evaluation resource observation scope must be nonempty",
                        code="run_writer.invalid_phase_summary",
                    )
                receipt["resource_observation_scope"] = resource_observation_scope
            if resource_high_water_observed_after_events is not None:
                receipt["resource_high_water_observed_after_events"] = _strict_mapping(
                    resource_high_water_observed_after_events,
                    field="evaluation resource observation",
                )

        _attach_rank_phase_receipt(
            receipt,
            rank_resources=rank_resources,
            rank_details=rank_details,
        )

        state = self.read_run()
        measurement = state["measurement"]
        if measurement["active_phase"] is not None:
            raise ArtifactContractError(
                "an aggregate summary cannot be recorded while a phase is active",
                code="run_writer.phase_already_active",
                context={"active_phase": measurement["active_phase"]},
            )
        if phase in measurement["phases"]:
            raise ArtifactContractError(
                "a run phase can be recorded only once",
                code="run_writer.phase_already_recorded",
                context={"phase": phase},
            )
        measurement["phases"][phase] = receipt
        measurement["phase_order"].append(phase)
        if phase == "steady_state":
            measurement["accepted_measured_steps"] = accepted
            measurement["expected_measured_steps"] = expected
            measurement["steady_state_eligible"] = bool(
                status == "completed" and expected > 0 and accepted == expected
            )
        if status == "failed":
            measurement["terminal_phase"] = phase
            measurement["terminal_phase_status"] = "failed"
            measurement["failure_phase"] = phase
            measurement["steady_state_eligible"] = False
        elif status == "completed":
            measurement["terminal_phase"] = phase
            measurement["terminal_phase_status"] = "completed"
            measurement["last_completed_phase"] = phase
        self._write_json_atomic(self.run_path, state)

    def finish_phase(
        self,
        phase: str,
        *,
        status: str,
        completed_at: str,
        duration_seconds: float,
        resources: Mapping[str, Any] | None = None,
        accepted_measured_steps: int | None = None,
        expected_measured_steps: int | None = None,
        rank_resources: Mapping[str, Any] | None = None,
        rank_details: Mapping[str, Any] | None = None,
    ) -> None:
        """Finish the active phase and update the process high-water receipt."""

        _validate_phase_name(phase)
        if status not in {"completed", "failed"}:
            raise ArtifactContractError(
                "phase terminal status must be completed or failed",
                code="run_writer.invalid_phase_status",
                context={"status": status},
            )
        _validate_timestamp(completed_at, field="completed_at")
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, (int, float))
            or not math.isfinite(float(duration_seconds))
            or float(duration_seconds) < 0.0
        ):
            raise ArtifactContractError(
                "phase duration must be finite and nonnegative",
                code="run_writer.invalid_phase_duration",
            )
        if accepted_measured_steps is not None:
            _validate_nonnegative_int(
                accepted_measured_steps, field="accepted_measured_steps"
            )
        if expected_measured_steps is not None:
            _validate_nonnegative_int(
                expected_measured_steps, field="expected_measured_steps"
            )
        if (
            phase == "steady_state"
            and status == "completed"
            and (
                accepted_measured_steps is None
                or expected_measured_steps is None
                or expected_measured_steps <= 0
            )
        ):
            raise ArtifactContractError(
                "completed steady state requires accepted and positive expected counts",
                code="run_writer.invalid_measured_step_count",
            )
        if phase != "steady_state" and (
            accepted_measured_steps is not None or expected_measured_steps is not None
        ):
            raise ArtifactContractError(
                "measured step count belongs only to the steady-state phase",
                code="run_writer.invalid_measured_step_count",
            )

        state = self.read_run()
        measurement = state["measurement"]
        if measurement["active_phase"] != phase:
            raise ArtifactContractError(
                "only the active phase can be finished",
                code="run_writer.phase_not_active",
                context={
                    "active_phase": measurement["active_phase"],
                    "requested_phase": phase,
                },
            )
        receipt = measurement["phases"].get(phase)
        if not isinstance(receipt, dict) or receipt.get("status") != "running":
            raise ArtifactContractError(
                "active phase receipt is inconsistent",
                code="run_writer.phase_state_invalid",
                context={"phase": phase},
            )
        receipt.update(
            status=status,
            completed_at=completed_at,
            duration_seconds=float(duration_seconds),
        )
        _attach_rank_phase_receipt(
            receipt,
            rank_resources=rank_resources,
            rank_details=rank_details,
        )
        measurement["active_phase"] = None
        measurement["terminal_phase"] = phase
        measurement["terminal_phase_status"] = status
        if status == "failed":
            measurement["failure_phase"] = phase
            measurement["steady_state_eligible"] = False
        else:
            measurement["last_completed_phase"] = phase
        if phase == "steady_state" and status == "completed":
            measurement["accepted_measured_steps"] = accepted_measured_steps
            measurement["expected_measured_steps"] = expected_measured_steps
            measurement["steady_state_eligible"] = bool(
                accepted_measured_steps == expected_measured_steps
            )
        _merge_resources(measurement, resources)
        self._write_json_atomic(self.run_path, state)

    def observe_resources(self, resources: Mapping[str, Any]) -> None:
        """Merge one process-local resource sample without changing phase state."""

        state = self.read_run()
        _merge_resources(state["measurement"], resources)
        self._write_json_atomic(self.run_path, state)

    def record_checkpoint_publication_event(
        self,
        *,
        step: int,
        status: str,
        started_at: str,
        completed_at: str,
        duration_seconds: float,
        is_final: bool,
        exact_training_state_enabled: bool,
        checkpoint_identity: Mapping[str, Any] | None,
        inference_payload_identity: Mapping[str, Any] | None,
        committed_progress: Mapping[str, Any] | None,
        failure_code: str | None,
    ) -> None:
        """Append one bounded receipt from the authoritative checkpoint publisher."""

        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ArtifactContractError(
                "checkpoint publication step must be a positive integer",
                code="run_writer.invalid_checkpoint_publication_event",
            )
        if status not in {"completed", "failed"}:
            raise ArtifactContractError(
                "checkpoint publication status must be completed or failed",
                code="run_writer.invalid_checkpoint_publication_event",
            )
        _validate_timestamp(started_at, field="started_at")
        _validate_timestamp(completed_at, field="completed_at")
        duration = _validate_nonnegative_finite(
            duration_seconds,
            field="checkpoint_publication.duration_seconds",
        )
        if not isinstance(is_final, bool) or not isinstance(
            exact_training_state_enabled, bool
        ):
            raise ArtifactContractError(
                "checkpoint publication flags must be boolean",
                code="run_writer.invalid_checkpoint_publication_event",
            )
        if status == "completed":
            if failure_code is not None:
                raise ArtifactContractError(
                    "a completed checkpoint publication cannot carry a failure code",
                    code="run_writer.invalid_checkpoint_publication_event",
                )
            if exact_training_state_enabled and checkpoint_identity is None:
                raise ArtifactContractError(
                    "a completed exact checkpoint requires its committed identity",
                    code="run_writer.invalid_checkpoint_publication_event",
                )
            if inference_payload_identity is None or committed_progress is None:
                raise ArtifactContractError(
                    "a completed checkpoint publication requires payload identity and progress",
                    code="run_writer.invalid_checkpoint_publication_event",
                )
        else:
            if checkpoint_identity is not None:
                raise ArtifactContractError(
                    "a failed checkpoint publication cannot claim a committed identity",
                    code="run_writer.invalid_checkpoint_publication_event",
                )
            if inference_payload_identity is not None or committed_progress is not None:
                raise ArtifactContractError(
                    "a failed checkpoint publication cannot claim payload identity or progress",
                    code="run_writer.invalid_checkpoint_publication_event",
                )
            if (
                not isinstance(failure_code, str)
                or not failure_code
                or len(failure_code) > 128
                or not failure_code.isascii()
            ):
                raise ArtifactContractError(
                    "a failed checkpoint publication requires a bounded failure code",
                    code="run_writer.invalid_checkpoint_publication_event",
                )
        identity = (
            None
            if checkpoint_identity is None
            else _checkpoint_publication_identity(
                checkpoint_identity,
                step=step,
                checkpoint_dir=self.checkpoints_dir / f"step-{step}",
            )
        )
        if not exact_training_state_enabled and identity is not None:
            raise ArtifactContractError(
                "an inference-only checkpoint cannot carry exact-state identity",
                code="run_writer.invalid_checkpoint_publication_event",
            )
        checkpoint_dir = self.checkpoints_dir / f"step-{step}"
        payload_identity = (
            None
            if inference_payload_identity is None
            else admit_inference_checkpoint_payload_identity(
                checkpoint_dir,
                inference_payload_identity,
            )
        )
        progress = (
            None
            if committed_progress is None
            else _checkpoint_committed_progress(committed_progress, step=step)
        )

        state = self.read_run()
        resolved_max_steps = state.get("resolved_max_steps")
        if isinstance(resolved_max_steps, int) and step > resolved_max_steps:
            raise ArtifactContractError(
                "checkpoint publication step exceeds the resolved schedule",
                code="run_writer.invalid_checkpoint_publication_event",
            )
        events = state["measurement"].setdefault("checkpoint_publication_events", [])
        if not isinstance(events, list) or len(events) >= min(
            _MAX_CHECKPOINT_PUBLICATION_EVENTS,
            resolved_max_steps
            if isinstance(resolved_max_steps, int)
            else _MAX_CHECKPOINT_PUBLICATION_EVENTS,
        ):
            raise ArtifactContractError(
                "checkpoint publication event inventory exceeds its bound",
                code="run_writer.checkpoint_publication_events_bounded",
            )
        if events and (
            not isinstance(events[-1], Mapping)
            or not isinstance(events[-1].get("step"), int)
            or int(events[-1]["step"]) >= step
        ):
            raise ArtifactContractError(
                "checkpoint publication steps must append in strictly increasing order",
                code="run_writer.invalid_checkpoint_publication_event",
            )
        events.append(
            {
                "schema": "coordexp-swift-checkpoint-publication-event",
                "schema_version": 2,
                "step": step,
                "status": status,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration,
                "duration_clock": "monotonic",
                "checkpoint_path": f"checkpoints/step-{step}",
                "is_final": is_final,
                "exact_training_state_enabled": exact_training_state_enabled,
                "checkpoint_identity": identity,
                "inference_payload_identity": payload_identity,
                "committed_progress": progress,
                "failure_code": failure_code,
            }
        )
        if progress is not None:
            state.update(
                completed_steps=progress["completed_steps"],
                consumed_packs=progress["consumed_packs"],
                checkpoint_event_count=len(events),
                final_optimizer_update_status=progress["optimizer_update_status"],
                final_finite_status=progress["finite_status"],
            )
        self._write_json_atomic(self.run_path, state)

    def finalize(
        self,
        *,
        status: str,
        updated_at: str,
        completed_steps: int,
        consumed_packs: int,
        checkpoint_event_count: int,
        optimizer_update_status: str | None,
        finite_status: str | None,
        terminal_error: str | None = None,
        entry_started_monotonic: float | None = None,
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ArtifactContractError(
                "terminal status must be completed or failed",
                code="run_writer.invalid_terminal_status",
            )
        state = self.read_run()
        _validate_finalization_checkpoint_progress(
            state,
            completed_steps=completed_steps,
            consumed_packs=consumed_packs,
            checkpoint_event_count=checkpoint_event_count,
            optimizer_update_status=optimizer_update_status,
            finite_status=finite_status,
        )
        state.update(
            {
                "status": status,
                "updated_at": updated_at,
                "completed_at": updated_at if status == "completed" else None,
                "completed_steps": completed_steps,
                "consumed_packs": consumed_packs,
                "checkpoint_event_count": checkpoint_event_count,
                "final_optimizer_update_status": optimizer_update_status,
                "final_finite_status": finite_status,
                "terminal_error": terminal_error[:1024] if terminal_error else None,
            }
        )
        if status == "failed":
            measurement = state["measurement"]
            measurement["steady_state_eligible"] = False
            if (
                measurement["active_phase"] is None
                and measurement["failure_phase"] is None
            ):
                measurement["phases"][_UNPHASED_FAILURE_PHASE] = {
                    "status": "failed",
                    "reason": "run_failed_outside_named_phase",
                    "started_at": None,
                    "completed_at": updated_at,
                    "duration_seconds": 0.0,
                }
                measurement["phase_order"].append(_UNPHASED_FAILURE_PHASE)
                measurement["terminal_phase"] = _UNPHASED_FAILURE_PHASE
                measurement["terminal_phase_status"] = "failed"
                measurement["failure_phase"] = _UNPHASED_FAILURE_PHASE
        self._write_json_atomic(self.run_path, state)
        if entry_started_monotonic is not None:
            _validate_nonnegative_finite(
                entry_started_monotonic,
                field="entry_started_monotonic",
            )
            measurement_completed_at = _utc_now()
            state["measurement"]["entry_to_terminal"] = {
                "status": "completed",
                "started_at": state["measurement"]["entry_to_terminal"]["started_at"],
                "completed_at": measurement_completed_at,
                "duration_seconds": max(
                    0.0, time.monotonic() - float(entry_started_monotonic)
                ),
                "clock": "monotonic",
                "boundary": (
                    "training_entry_to_terminal_state_durable_before_"
                    "measurement_annotation"
                ),
            }
            state["updated_at"] = measurement_completed_at
            if status == "completed":
                state["completed_at"] = measurement_completed_at
            self._write_json_atomic(self.run_path, state)

    def write_final(self, *, step: int) -> Path:
        path = self._checkpoint_path(step)
        alias = {"step": step, "checkpoint_path": path}
        output = self.checkpoints_dir / "final.json"
        self._write_json_atomic(output, alias)
        return output

    def write_best(
        self,
        *,
        step: int,
        selector: str,
        value: float,
        optimizer_update_status: str,
        finite_status: str,
        checkpoint_committed: bool,
        allow_unsafe_override: bool = False,
    ) -> bool:
        eligible = (
            checkpoint_committed
            and optimizer_update_status == "applied"
            and finite_status == "finite"
            and math.isfinite(value)
        )
        if not eligible and not allow_unsafe_override:
            return False
        if not math.isfinite(value):
            return False
        output = self.checkpoints_dir / "best.json"
        if output.exists():
            current = json.loads(output.read_text(encoding="utf-8"))
            if value <= current["value"]:
                return False
        payload = {
            "step": step,
            "checkpoint_path": self._checkpoint_path(step),
            "selector": selector,
            "value": value,
        }
        if allow_unsafe_override and not eligible:
            payload["unsafe_override"] = True
        self._write_json_atomic(output, payload)
        return True

    def file_inventory(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                path.relative_to(self.run_dir).as_posix()
                for path in self.run_dir.rglob("*")
                if path.is_file()
            )
        )

    def _checkpoint_path(self, step: int) -> str:
        if step <= 0:
            raise ArtifactContractError(
                "checkpoint step must be positive",
                code="run_writer.invalid_checkpoint_step",
            )
        return f"checkpoints/step-{step}"

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        encoded = _strict_json_dumps(payload) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            _fsync_directory(path.parent)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


def admit_exact_resume_checkpoint_publication(
    checkpoint_dir: str | Path,
    *,
    checkpoint_step: int,
    training_state_manifest_file_sha256: str,
    training_state_aggregate_digest: str,
    parent_run_id: str,
    parent_segment_id: str,
) -> dict[str, Any]:
    """Authenticate one exact checkpoint against its parent run commit event."""

    try:
        raw_checkpoint_dir = Path(checkpoint_dir).expanduser()
        if raw_checkpoint_dir.is_symlink():
            raise ValueError("checkpoint directory cannot be a symlink")
        resolved_checkpoint_dir = raw_checkpoint_dir.resolve(strict=True)
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step <= 0
            or resolved_checkpoint_dir.name != f"step-{checkpoint_step}"
            or resolved_checkpoint_dir.parent.name != "checkpoints"
        ):
            raise ValueError("checkpoint path and step disagree")
        run_dir = resolved_checkpoint_dir.parent.parent
        run_path = run_dir / "run.json"
        if run_path.is_symlink() or not run_path.is_file():
            raise ValueError("parent run record is missing or not a regular file")
        state = json.loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise ValueError("parent run record must be a JSON object")
        _strict_json_dumps(state)
        if state.get("run_id") != parent_run_id:
            raise ValueError("parent run identity disagrees with training state")
        continuation = state.get("continuation")
        if (
            not isinstance(continuation, Mapping)
            or continuation.get("segment_id") != parent_segment_id
        ):
            raise ValueError("parent segment identity disagrees with training state")
        if state.get("run_dir") != str(run_dir.resolve()):
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
        admitted_payload_identity = admit_inference_checkpoint_payload_identity(
            resolved_checkpoint_dir,
            event["inference_payload_identity"],
        )
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
    except Exception as exc:
        if isinstance(exc, ArtifactContractError) and exc.code == (
            "run_writer.exact_resume_publication_invalid"
        ):
            raise
        raise ArtifactContractError(
            "exact-resume checkpoint lacks one authoritative parent publication event",
            code="run_writer.exact_resume_publication_invalid",
            context={
                "checkpoint_dir": str(Path(checkpoint_dir).expanduser()),
                "checkpoint_step": checkpoint_step,
                "observed_error_code": getattr(exc, "code", type(exc).__name__),
            },
            cause=exc,
        ) from exc


def _fsync_directory(path: Path) -> None:
    """Durably publish directory-entry mutations owned by this writer."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _normalize_logging_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("split") not in {"train", "eval"}:
        raise ArtifactContractError(
            "logging split must be train or eval",
            code="run_writer.invalid_logging_split",
        )
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ArtifactContractError(
            "logging step must be a positive integer",
            code="run_writer.invalid_logging_step",
        )
    normalized = deepcopy(dict(row))
    fields: list[str] = []
    normalized = _replace_non_finite(normalized, path="", fields=fields)
    declared = normalized.get("non_finite_fields", [])
    if not isinstance(declared, list) or not all(
        isinstance(item, str) for item in declared
    ):
        raise ArtifactContractError(
            "non_finite_fields must be a list of field names",
            code="run_writer.invalid_non_finite_fields",
        )
    normalized["non_finite_fields"] = sorted(set(declared).union(fields))
    _reject_non_finite(normalized)
    return normalized


def _provenance_payload(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if provenance is not None:
        return _strict_mapping(provenance, field="provenance")
    unavailable = {"status": "unavailable", "reason": "not_collected_by_caller"}
    return {
        "schema_version": 1,
        "repository": {
            "commit": dict(unavailable),
            "state": "unavailable",
            "tracked_changes_present": None,
            "untracked_changes_present": None,
            "execution_relevant_changes": {
                "count": 0,
                "path_classes": {},
                "truncated": False,
            },
            "execution_relevant_digest": dict(unavailable),
        },
        "dependencies": {},
        "runtime": {},
    }


def _continuation_payload(
    *,
    run_id: str,
    segment_id: str,
    lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if lineage is None:
        return {
            "schema_version": 1,
            "segment_id": segment_id,
            "continuation_index": 0,
            "parent": None,
        }
    if not isinstance(lineage, Mapping) or set(lineage) != _CONTINUATION_LINEAGE_FIELDS:
        _invalid_continuation_lineage(
            "continuation lineage must contain the complete exact field set"
        )
    parent_run_id = _validate_lineage_identity(
        lineage["parent_run_id"], field="parent_run_id"
    )
    parent_segment_id = _validate_lineage_identity(
        lineage["parent_segment_id"], field="parent_segment_id"
    )
    if parent_run_id == run_id or parent_segment_id == segment_id:
        _invalid_continuation_lineage(
            "a continuation cannot identify itself as its parent"
        )
    parent_index = _validate_lineage_index(
        lineage["parent_continuation_index"],
        field="parent_continuation_index",
    )
    continuation_index = _validate_lineage_index(
        lineage["continuation_index"], field="continuation_index"
    )
    if continuation_index <= 0 or continuation_index != parent_index + 1:
        _invalid_continuation_lineage(
            "continuation index must advance its parent index exactly once"
        )
    checkpoint_identity = _validate_parent_checkpoint_identity(
        lineage["parent_checkpoint_identity"]
    )
    return {
        "schema_version": 1,
        "segment_id": segment_id,
        "continuation_index": continuation_index,
        "parent": {
            "run_id": parent_run_id,
            "segment_id": parent_segment_id,
            "checkpoint_identity": checkpoint_identity,
            "continuation_index": parent_index,
        },
    }


def _validate_segment_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not value.isascii()
    ):
        raise ArtifactContractError(
            "segment identity must be a nonempty bounded ASCII string",
            code="run_writer.invalid_segment_id",
        )
    return value


def _validate_lineage_identity(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not value.isascii()
    ):
        _invalid_continuation_lineage(
            f"{field} must be a nonempty bounded ASCII string"
        )
    return value


def _validate_lineage_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid_continuation_lineage(f"{field} must be a nonnegative integer")
    return value


def _validate_parent_checkpoint_identity(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _PARENT_CHECKPOINT_IDENTITY_FIELDS
    ):
        _invalid_continuation_lineage(
            "parent checkpoint identity must contain the exact resolved path and digest fields"
        )
    resolved_path = value["resolved_path"]
    if (
        not isinstance(resolved_path, str)
        or not resolved_path
        or len(resolved_path) > 4096
        or not Path(resolved_path).is_absolute()
        or str(Path(resolved_path).resolve()) != resolved_path
    ):
        _invalid_continuation_lineage(
            "parent checkpoint identity path must be absolute and resolved"
        )
    checkpoint_step = value["checkpoint_step"]
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step <= 0
    ):
        _invalid_continuation_lineage(
            "parent checkpoint step must be a positive integer"
        )
    manifest_digest = value["training_state_manifest_file_sha256"]
    aggregate_digest = value["training_state_aggregate_digest"]
    if not _is_lowercase_sha256(manifest_digest) or not _is_lowercase_sha256(
        aggregate_digest
    ):
        _invalid_continuation_lineage(
            "parent training-state identities must be lowercase SHA-256 digests"
        )
    return {
        "resolved_path": resolved_path,
        "checkpoint_step": checkpoint_step,
        "training_state_manifest_file_sha256": manifest_digest,
        "training_state_aggregate_digest": aggregate_digest,
    }


def _is_lowercase_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _invalid_continuation_lineage(message: str) -> None:
    raise ArtifactContractError(
        message,
        code="run_writer.invalid_continuation_lineage",
    )


def _checkpoint_publication_identity(
    value: Mapping[str, Any],
    *,
    step: int,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    fields = _PARENT_CHECKPOINT_IDENTITY_FIELDS
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ArtifactContractError(
            "checkpoint publication identity must contain the complete exact field set",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    resolved_path = value["resolved_path"]
    if (
        not isinstance(resolved_path, str)
        or resolved_path != str(checkpoint_dir.resolve())
        or value["checkpoint_step"] != step
        or not _is_lowercase_sha256(value["training_state_manifest_file_sha256"])
        or not _is_lowercase_sha256(value["training_state_aggregate_digest"])
    ):
        raise ArtifactContractError(
            "checkpoint publication identity disagrees with its committed checkpoint",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    return {
        "checkpoint_step": step,
        "resolved_path": resolved_path,
        "training_state_aggregate_digest": value["training_state_aggregate_digest"],
        "training_state_manifest_file_sha256": value[
            "training_state_manifest_file_sha256"
        ],
    }


def _checkpoint_committed_progress(
    value: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _COMMITTED_PROGRESS_FIELDS:
        raise ArtifactContractError(
            "checkpoint committed progress must contain the complete exact field set",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    completed_steps = value["completed_steps"]
    consumed_packs = value["consumed_packs"]
    if (
        value["schema"] != "coordexp-swift-checkpoint-committed-progress"
        or value["schema_version"] != 1
        or isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps != step
        or isinstance(consumed_packs, bool)
        or not isinstance(consumed_packs, int)
        or consumed_packs < 0
    ):
        raise ArtifactContractError(
            "checkpoint committed progress disagrees with its publication step",
            code="run_writer.invalid_checkpoint_publication_event",
        )
    for field in ("optimizer_update_status", "finite_status"):
        status = value[field]
        if status is not None and (
            not isinstance(status, str)
            or not status
            or len(status) > 128
            or not status.isascii()
        ):
            raise ArtifactContractError(
                "checkpoint committed progress status is invalid",
                code="run_writer.invalid_checkpoint_publication_event",
                context={"field": field},
            )
    return {
        "schema": "coordexp-swift-checkpoint-committed-progress",
        "schema_version": 1,
        "completed_steps": completed_steps,
        "consumed_packs": consumed_packs,
        "optimizer_update_status": value["optimizer_update_status"],
        "finite_status": value["finite_status"],
    }


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


def _measurement_payload(
    context: Mapping[str, Any] | None,
    *,
    entry_started_at: str,
) -> dict[str, Any]:
    _validate_timestamp(entry_started_at, field="entry_started_at")
    payload = (
        {
            "comparison_arm": "unclassified",
            "wall_clock_scope": "training_entry_to_terminal_artifact",
            "warmup_exclusion_steps": None,
        }
        if context is None
        else _strict_mapping(context, field="measurement context")
    )
    return {
        "schema_version": 1,
        "context": payload,
        "active_phase": None,
        "terminal_phase": None,
        "terminal_phase_status": None,
        "last_completed_phase": None,
        "failure_phase": None,
        "phase_order": [],
        "phases": {},
        "checkpoint_publication_events": [],
        "resource_high_water": None,
        "accepted_measured_steps": 0,
        "expected_measured_steps": None,
        "steady_state_eligible": False,
        "entry_to_terminal": {
            "status": "running",
            "started_at": entry_started_at,
            "completed_at": None,
            "duration_seconds": None,
            "clock": "monotonic",
            "boundary": (
                "training_entry_to_terminal_state_durable_before_measurement_annotation"
            ),
        },
    }


def _attach_rank_phase_receipt(
    receipt: dict[str, Any],
    *,
    rank_resources: Mapping[str, Any] | None,
    rank_details: Mapping[str, Any] | None,
) -> None:
    if rank_resources is None:
        if rank_details is not None:
            raise ArtifactContractError(
                "rank details require a rank resource receipt",
                code="run_writer.invalid_additive_receipt",
                context={"field": "rank_details"},
            )
        return
    resources = validate_rank_cpu_resource_receipt(rank_resources)
    receipt["rank_resources"] = resources
    if rank_details is None:
        return
    details = _strict_mapping(rank_details, field="rank details")
    expected_ranks = {str(rank) for rank in range(int(resources["world_size"]))}
    if set(details) != expected_ranks:
        raise ArtifactContractError(
            "rank details must cover every resource rank exactly once",
            code="run_writer.invalid_additive_receipt",
            context={"field": "rank_details"},
        )
    _validate_bounded_rank_detail_value(details, depth=0)
    encoded = _strict_json_dumps(details, compact=True).encode("utf-8")
    if len(encoded) > 32 * 1024:
        raise ArtifactContractError(
            "rank details exceed the bounded artifact size",
            code="run_writer.invalid_additive_receipt",
            context={"field": "rank_details"},
        )
    receipt["rank_details"] = details


def _validate_bounded_rank_detail_value(value: Any, *, depth: int) -> None:
    if depth > 6:
        raise ArtifactContractError(
            "rank details exceed the bounded nesting depth",
            code="run_writer.invalid_additive_receipt",
            context={"field": "rank_details"},
        )
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
    elif isinstance(value, str):
        if len(value) <= 256 and value.isascii():
            return
    elif isinstance(value, Mapping):
        if len(value) <= 128 and all(
            isinstance(key, str) and 0 < len(key) <= 64 and key.isascii()
            for key in value
        ):
            for item in value.values():
                _validate_bounded_rank_detail_value(item, depth=depth + 1)
            return
    elif isinstance(value, (list, tuple)) and len(value) <= 128:
        for item in value:
            _validate_bounded_rank_detail_value(item, depth=depth + 1)
        return
    raise ArtifactContractError(
        "rank details contain an unbounded or unsupported value",
        code="run_writer.invalid_additive_receipt",
        context={"field": "rank_details"},
    )


def _strict_mapping(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(
            f"{field} must be a mapping",
            code="run_writer.invalid_additive_receipt",
            context={"field": field},
        )
    payload = deepcopy(dict(value))
    _strict_json_dumps(payload)
    return payload


def _validate_phase_name(phase: str) -> None:
    if phase not in _PHASE_NAMES:
        raise ArtifactContractError(
            "run phase name is not supported",
            code="run_writer.invalid_phase_name",
            context={"phase": phase},
        )


def _validate_timestamp(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactContractError(
            "phase timestamps must be nonempty strings",
            code="run_writer.invalid_phase_timestamp",
            context={"field": field},
        )


def _validate_nonnegative_finite(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ArtifactContractError(
            f"{field} must be finite and nonnegative",
            code="run_writer.invalid_phase_summary",
            context={"field": field},
        )
    return float(value)


def _validate_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactContractError(
            f"{field} must be a nonnegative integer",
            code="run_writer.invalid_measured_step_count",
            context={"field": field},
        )
    return value


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


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


def _replace_non_finite(value: Any, *, path: str, fields: list[str]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        fields.append(path)
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _replace_non_finite(
                item, path=f"{path}.{key}" if path else str(key), fields=fields
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _replace_non_finite(item, path=f"{path}[{index}]", fields=fields)
            for index, item in enumerate(value)
        ]
    return value


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactContractError(
            "non-finite value remains after logging normalization",
            code="run_writer.non_finite_remains",
        )
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def _strict_json_dumps(payload: Any, *, compact: bool = False) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":") if compact else None,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "artifact payload is not strict-JSON serializable",
            code="run_writer.not_json_serializable",
            cause=exc,
        ) from exc
