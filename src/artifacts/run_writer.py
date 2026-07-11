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
        if code not in counts and len(counts) >= _MAX_WARNING_CODES - reserved_other_slot:
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
        encoded = _strict_json_dumps(payload) + "\n"
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
    if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
        raise ArtifactContractError(
            "non_finite_fields must be a list of field names",
            code="run_writer.invalid_non_finite_fields",
        )
    normalized["non_finite_fields"] = sorted(set(declared).union(fields))
    _reject_non_finite(normalized)
    return normalized


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
