"""Concrete single-writer artifacts for one CoordExp training run."""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts.json_values import strict_json_text
from src.common.errors import ArtifactContractError


_MAX_WARNING_CODES = 32
_MAX_WARNING_COUNT = 2**31 - 1
_OTHER_WARNING_CODE = "other"


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
    ) -> "RunWriter":
        if world_size <= 0 or (
            resolved_max_steps is not None and resolved_max_steps <= 0
        ):
            raise ArtifactContractError(
                "world size and resolved maximum steps must be positive",
                code="run_writer.invalid_runtime_summary",
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
                },
            )
            staging_writer.logging_path.touch(exist_ok=False)
            os.replace(staging_dir, run_dir)
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
        encoded = _run_writer_json_text(normalized, compact=True)
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

    def bind_rollout_calibration(
        self,
        *,
        bank_identity: str,
        bank_fingerprint: str,
        source_composite_fingerprint: str,
        profile: str,
        records_sha256: str,
        record_count: int,
        split_counts: Mapping[str, int],
        event_counts: Mapping[str, int],
        rejection_reasons: Mapping[str, int],
        trajectory_source_checkpoint: Mapping[str, Any] | None = None,
        training_warm_start_checkpoint: Mapping[str, Any] | None = None,
        off_policy_state_bank_replay: bool = False,
    ) -> None:
        """Bind one immutable rollout-calibration provenance summary to run.json."""
        string_fields = {
            "bank_identity": bank_identity,
            "bank_fingerprint": bank_fingerprint,
            "source_composite_fingerprint": source_composite_fingerprint,
            "profile": profile,
            "records_sha256": records_sha256,
        }
        invalid_fields = tuple(
            name
            for name, value in string_fields.items()
            if not isinstance(value, str) or not value.strip()
        )
        if invalid_fields:
            raise ArtifactContractError(
                "rollout-calibration identity fields must be nonempty strings",
                code="run_writer.invalid_rollout_calibration_binding",
                context={"invalid_fields": invalid_fields},
            )
        binding = {
            **string_fields,
            "validation_receipt": {
                "status": "validated",
                "bank_id": bank_identity,
                "source_checkpoint_id": source_composite_fingerprint,
                "records_sha256": records_sha256,
                "record_count": _normalize_binding_count(
                    record_count,
                    field="record_count",
                ),
                "split_counts": _normalize_binding_counts(
                    split_counts,
                    field="split_counts",
                ),
                "event_family_counts": _normalize_binding_counts(
                    event_counts,
                    field="event_counts",
                ),
                "rejection_reasons": _normalize_binding_counts(
                    rejection_reasons,
                    field="rejection_reasons",
                ),
            },
            "split_counts": _normalize_binding_counts(
                split_counts, field="split_counts"
            ),
            "event_counts": _normalize_binding_counts(
                event_counts, field="event_counts"
            ),
        }
        if (trajectory_source_checkpoint is None) != (
            training_warm_start_checkpoint is None
        ):
            raise ArtifactContractError(
                "rollout-calibration checkpoint replay evidence must provide both identities",
                code="run_writer.invalid_rollout_calibration_checkpoint_replay",
            )
        if trajectory_source_checkpoint is not None:
            binding["checkpoint_replay"] = {
                "trajectory_source_checkpoint": dict(trajectory_source_checkpoint),
                "training_warm_start_checkpoint": dict(
                    training_warm_start_checkpoint or {}
                ),
                "off_policy_state_bank_replay": bool(
                    off_policy_state_bank_replay
                ),
            }
        state = self.read_run()
        if "rollout_calibration" in state:
            raise ArtifactContractError(
                "rollout-calibration run binding is immutable",
                code="run_writer.rollout_calibration_already_bound",
            )
        state["rollout_calibration"] = binding
        self._write_json_atomic(self.run_path, state)

    def bind_rollout_calibration_qualification(
        self,
        *,
        source_checkpoint: Mapping[str, Any],
        source_step_zero_parity: Mapping[str, Any],
        trainable_surface: Mapping[str, Any],
    ) -> None:
        state = self.read_run()
        calibration = state.get("rollout_calibration")
        if not isinstance(calibration, dict):
            raise ArtifactContractError(
                "rollout-calibration identity must be bound before qualification evidence",
                code="run_writer.rollout_calibration_binding_missing",
            )
        if "qualification" in calibration:
            raise ArtifactContractError(
                "rollout-calibration qualification binding is immutable",
                code="run_writer.rollout_calibration_qualification_already_bound",
            )
        calibration["qualification"] = {
            "source_checkpoint": dict(source_checkpoint),
            "source_step_zero_parity": dict(source_step_zero_parity),
            "trainable_surface": dict(trainable_surface),
            "post_backward_gradient": {
                "status": "not_run",
                "reason": "no_completed_calibration_backward",
            },
            "target_margins": {
                "status": "not_run",
                "pre_update": {},
                "post_update": {},
                "reason": "shared_real_smoke_fixture_not_available",
            },
            "checkpoint_reload_and_ordinary_inference": {
                "status": "not_run",
                "reason": "shared_real_smoke_fixture_not_available",
            },
        }
        state["rollout_calibration"] = calibration
        self._write_json_atomic(self.run_path, state)

    def record_rollout_calibration_step_evidence(
        self,
        *,
        planned_step_id: int,
        post_backward_gradient: Mapping[str, Any],
        pre_update_margins: Mapping[str, float],
        post_update_margins: Mapping[str, float] | None = None,
        optimizer_update_status: str,
    ) -> None:
        state = self.read_run()
        calibration = state.get("rollout_calibration")
        qualification = (
            calibration.get("qualification") if isinstance(calibration, dict) else None
        )
        if not isinstance(qualification, dict):
            raise ArtifactContractError(
                "rollout-calibration qualification must be bound before step evidence",
                code="run_writer.rollout_calibration_qualification_missing",
            )
        qualification["post_backward_gradient"] = {
            **dict(post_backward_gradient),
            "planned_step_id": _normalize_binding_count(
                planned_step_id, field="planned_step_id"
            ),
            "optimizer_update_status": str(optimizer_update_status),
        }
        normalized_post_update = (
            {
                str(name): float(value)
                for name, value in sorted(post_update_margins.items())
            }
            if post_update_margins is not None
            else {}
        )
        normalized_pre_update = {
            str(name): float(value)
            for name, value in sorted(pre_update_margins.items())
        }
        if normalized_post_update and set(normalized_post_update) != set(
            normalized_pre_update
        ):
            raise ArtifactContractError(
                "post-update calibration margin keys differ from pre-update keys",
                code="run_writer.rollout_calibration_margin_keys_mismatch",
                context={
                    "pre_update_keys": sorted(normalized_pre_update),
                    "post_update_keys": sorted(normalized_post_update),
                },
            )
        qualification["target_margins"] = {
            "status": "complete" if normalized_post_update else "partial",
            "pre_update": normalized_pre_update,
            "post_update": normalized_post_update,
            "reason": (
                "post_update_margin_replayed_without_event_reconsumption"
                if normalized_post_update
                else "post_update_margin_requires_shared_smoke_replay"
            ),
        }
        calibration["qualification"] = qualification
        state["rollout_calibration"] = calibration
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
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ArtifactContractError(
                "terminal status must be completed or failed",
                code="run_writer.invalid_terminal_status",
            )
        state = self.read_run()
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
        encoded = _run_writer_json_text(payload) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise


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


def _normalize_binding_counts(
    counts: Mapping[str, int],
    *,
    field: str,
) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        raise ArtifactContractError(
            "rollout-calibration counts must be mappings",
            code="run_writer.invalid_rollout_calibration_binding",
            context={"field": field, "value_type": type(counts).__name__},
        )
    normalized: dict[str, int] = {}
    for name, count in counts.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ArtifactContractError(
                "rollout-calibration count names and values must be valid",
                code="run_writer.invalid_rollout_calibration_binding",
                context={"field": field, "name": name, "count": count},
            )
        normalized[name] = count
    return dict(sorted(normalized.items()))


def _normalize_binding_count(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactContractError(
            "rollout-calibration binding counts must be nonnegative integers",
            code="run_writer.invalid_rollout_calibration_binding",
            context={"field": field, "value": value},
        )
    return int(value)


def _replace_non_finite(value: Any, *, path: str, fields: list[str]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        fields.append(path)
        return None
    if isinstance(value, Mapping):
        return {
            key: _replace_non_finite(
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


def _run_writer_json_text(payload: Any, *, compact: bool = False) -> str:
    """Keep the established run-writer error code while sharing validation."""

    try:
        return strict_json_text(payload, compact=compact)
    except ArtifactContractError as exc:
        raise ArtifactContractError(
            "artifact payload is not strict-JSON serializable",
            code="run_writer.not_json_serializable",
            cause=exc,
        ) from exc
