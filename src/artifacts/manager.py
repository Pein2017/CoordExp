"""Run artifact manager and manifest writer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.common.errors import ArtifactContractError
from src.artifacts.metric_stream import MetricStreamEvent
from src.config.models import ResolvedConfigArtifacts, ResolvedTrainConfig, RunDirectory
from src.config.writer import write_resolved_config_artifacts
from src.training.schedule import ResolvedStepSchedule, write_resolved_step_schedule


MANIFEST_NAME = "run_manifest.json"


@dataclass(frozen=True)
class RunArtifactManager:
    run_directory: RunDirectory
    run_id: str
    manifest_path: Path

    @classmethod
    def initialize(
        cls,
        *,
        run_directory: RunDirectory,
        run_id: str,
        created_at: str,
        runtime_identity: Mapping[str, Any],
        backend_status: Mapping[str, Any],
    ) -> "RunArtifactManager":
        run_dir = run_directory.run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ArtifactContractError(
                "run artifact directory must be empty before manifest initialization",
                code="artifact.run_dir_not_empty",
                context={"run_dir": str(run_dir)},
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / MANIFEST_NAME
        if manifest_path.exists():
            raise ArtifactContractError(
                "run manifest already exists",
                code="artifact.manifest_exists",
                context={"path": str(manifest_path)},
            )
        manager = cls(
            run_directory=RunDirectory(
                run_name=run_directory.run_name,
                artifact_root=run_directory.artifact_root.resolve(),
                run_dir=run_dir,
                collision_policy=run_directory.collision_policy,
            ),
            run_id=run_id,
            manifest_path=manifest_path,
        )
        manager._write_manifest(
            {
                "run_id": run_id,
                "run_name": run_directory.run_name,
                "run_dir": str(run_dir),
                "status": "initialized",
                "created_at": created_at,
                "updated_at": created_at,
                "completed_at": None,
                "configs": {},
                "resolution": {},
                "runtime_identity": dict(runtime_identity),
                "schedule": {},
                "receipts": {},
                "reports": {},
                "metrics": {"streams": {}},
                "checkpoints": {"items": [], "aliases": {}},
                "eval": {"forward": {}},
                "runtime": {
                    "artifact_root": str(run_directory.artifact_root.resolve()),
                    "collision_policy": run_directory.collision_policy,
                },
                "backend_status": _jsonable_mapping(backend_status),
                "warnings": [],
            }
        )
        return manager

    @property
    def run_dir(self) -> Path:
        return self.run_directory.run_dir

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def relative_artifact_path(self, path: Path) -> str:
        return self._relative_path(path)

    def write_resolved_config(self, resolved_config: ResolvedTrainConfig) -> Any:
        artifacts = self._write_or_reuse_resolved_config(resolved_config)
        resolved_payload = resolved_config.to_artifact_dict()
        manifest = self.read_manifest()
        manifest["configs"].update(
            {
                "resolved_yaml": self._relative_path(artifacts.yaml_path),
                "resolved_json": self._relative_path(artifacts.json_path),
            }
        )
        manifest["resolution"] = resolved_payload["resolution"]
        self._write_manifest(manifest)
        return artifacts

    def write_schedule(self, schedule: ResolvedStepSchedule) -> Path:
        output_path = self._write_or_reuse_schedule(schedule)
        manifest = self.read_manifest()
        manifest["schedule"] = {
            "resolved_step_schedule": self._relative_path(output_path),
            "resolved_max_steps": schedule.resolved_max_steps,
            "tail_fill_pack_count": schedule.tail_fill_pack_count,
            "runtime_batch": schedule.runtime_batch.to_artifact_dict(),
        }
        self._write_manifest(manifest)
        return output_path

    def write_report(
        self,
        name: str,
        payload: Mapping[str, Any],
    ) -> Path:
        _validate_name(name, field="report name")
        output_path = self.run_dir / "reports" / f"{name}.json"
        report_payload = dict(payload)
        if output_path.exists() and not _json_file_matches(output_path, report_payload):
            raise ArtifactContractError(
                "run report already exists",
                code="artifact.report_exists",
                context={"path": str(output_path)},
            )
        if not output_path.exists():
            _write_json(output_path, report_payload)
        manifest = self.read_manifest()
        manifest.setdefault("reports", {})[name] = self._relative_path(output_path)
        self._write_manifest(manifest)
        return output_path

    def write_receipt(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        category: str,
    ) -> Path:
        _validate_name(name, field="receipt name")
        _validate_name(category, field="receipt category")
        output_path = self.run_dir / "receipts" / category / f"{name}.json"
        receipt_payload = dict(payload)
        if output_path.exists() and not _json_file_matches(output_path, receipt_payload):
            raise ArtifactContractError(
                "subsystem receipt already exists",
                code="artifact.receipt_exists",
                context={"path": str(output_path)},
            )
        if not output_path.exists():
            _write_json(output_path, receipt_payload)
        manifest = self.read_manifest()
        category_receipts = manifest["receipts"].setdefault(category, {})
        category_receipts[name] = self._relative_path(output_path)
        self._write_manifest(manifest)
        return output_path

    def append_metric_event(self, event: MetricStreamEvent) -> Path:
        record = event.to_record()
        split = record["split"]
        _validate_metric_split(split)
        output_path = self.run_dir / "metrics" / f"{split}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.read_manifest()
        relative_path = self._relative_path(output_path)
        existing_stream = manifest["metrics"]["streams"].get(split)
        if existing_stream is not None and existing_stream != relative_path:
            raise ArtifactContractError(
                "metric stream is already registered at a different path",
                code="metric.stream_path_conflict",
                context={
                    "split": split,
                    "existing": existing_stream,
                    "new": relative_path,
                },
            )
        if existing_stream is None:
            output_path.touch(exist_ok=True)
            manifest["metrics"]["streams"][split] = relative_path
            self._write_manifest(manifest)
        if _metric_event_already_recorded(output_path, record):
            return output_path
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(_jsonl_dumps(record) + "\n")
        return output_path

    def write_eval_forward_summary(
        self,
        *,
        planned_step_id: int,
        summary: Mapping[str, Any],
    ) -> Path:
        if planned_step_id <= 0:
            raise ArtifactContractError(
                "eval.forward planned_step_id must be positive",
                code="eval_forward.planned_step_id",
                context={"planned_step_id": planned_step_id},
            )
        output_path = self.run_dir / "eval" / "forward" / f"step-{planned_step_id}.json"
        summary_payload = dict(summary)
        if output_path.exists():
            if not _json_file_matches(output_path, summary_payload):
                raise ArtifactContractError(
                    "eval.forward summary already exists with different content",
                    code="eval_forward.summary_exists",
                    context={"path": str(output_path)},
                )
        else:
            _write_json(output_path, summary_payload)
        manifest = self.read_manifest()
        manifest["eval"].setdefault("forward", {})
        manifest["eval"]["forward"][f"step-{planned_step_id}"] = self._relative_path(
            output_path
        )
        self._write_manifest(manifest)
        return output_path

    def register_checkpoint(
        self,
        *,
        checkpoint_id: str,
        planned_step_id: int,
        metadata_path: Path,
        final_alias_path: Path | None = None,
        best_alias_path: Path | None = None,
        best_acc_top1: Mapping[str, Any] | None = None,
    ) -> None:
        manifest = self.read_manifest()
        metadata_relative_path = self._relative_path(metadata_path)
        items = manifest["checkpoints"].setdefault("items", [])
        if metadata_relative_path not in items:
            items.append(metadata_relative_path)
        manifest["checkpoints"].setdefault("latest", {})
        manifest["checkpoints"]["latest"] = {
            "checkpoint_id": checkpoint_id,
            "planned_step_id": planned_step_id,
            "metadata_path": metadata_relative_path,
        }
        aliases = manifest["checkpoints"].setdefault("aliases", {})
        if final_alias_path is not None:
            aliases["final"] = self._relative_path(final_alias_path)
        if best_alias_path is not None and best_acc_top1 is not None:
            aliases["best_acc_top1"] = self._relative_path(best_alias_path)
            manifest["checkpoints"]["best_acc_top1"] = dict(best_acc_top1)
        self._write_manifest(manifest)

    def record_warning(
        self,
        *,
        code: str,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        manifest = self.read_manifest()
        manifest["warnings"].append(
            {
                "code": code,
                "message": message,
                "context": dict(context or {}),
            }
        )
        self._write_manifest(manifest)

    def finalize(self, *, status: str, completed_at: str) -> None:
        if status not in {"completed", "failed"}:
            raise ArtifactContractError(
                "run manifest final status must be completed or failed",
                code="artifact.final_status",
                context={"status": status},
            )
        manifest = self.read_manifest()
        manifest["status"] = status
        manifest["updated_at"] = completed_at
        manifest["completed_at"] = completed_at
        self._write_manifest(manifest)

    def _relative_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.run_dir)
        except ValueError as exc:
            raise ArtifactContractError(
                "artifact path must stay under run directory",
                code="artifact.path_escape",
                context={"path": str(resolved), "run_dir": str(self.run_dir)},
                cause=exc,
            ) from exc
        return relative.as_posix()

    def _write_manifest(self, payload: Mapping[str, Any]) -> None:
        _write_json(self.manifest_path, dict(payload))

    def _write_or_reuse_resolved_config(
        self,
        resolved_config: ResolvedTrainConfig,
    ) -> ResolvedConfigArtifacts:
        config_dir = self.run_dir / "configs"
        json_path = config_dir / "resolved.json"
        yaml_path = config_dir / "resolved.yaml"
        expected_payload = resolved_config.to_artifact_dict()
        if json_path.exists() or yaml_path.exists():
            if not json_path.exists() or not yaml_path.exists():
                raise ArtifactContractError(
                    "resolved config artifacts are partially materialized",
                    code="artifact.resolved_config_partial",
                    context={
                        "json_path": str(json_path),
                        "yaml_path": str(yaml_path),
                    },
                )
            if not _json_file_matches(json_path, expected_payload):
                raise ArtifactContractError(
                    "resolved config artifact already exists with different content",
                    code="artifact.resolved_config_exists",
                    context={"path": str(json_path)},
                )
            if not _yaml_file_matches(yaml_path, expected_payload):
                raise ArtifactContractError(
                    "resolved config artifact already exists with different content",
                    code="artifact.resolved_config_exists",
                    context={"path": str(yaml_path)},
                )
            return ResolvedConfigArtifacts(
                yaml_path=yaml_path,
                json_path=json_path,
                fingerprint=resolved_config.fingerprint,
            )
        return write_resolved_config_artifacts(
            resolved_config,
            self.run_dir,
            overwrite=False,
        )

    def _write_or_reuse_schedule(self, schedule: ResolvedStepSchedule) -> Path:
        output_path = self.run_dir / "resolved_step_schedule.json"
        expected_payload = schedule.to_artifact_dict()
        if output_path.exists():
            if not _json_file_matches(output_path, expected_payload):
                raise ArtifactContractError(
                    "resolved step schedule already exists with different content",
                    code="artifact.schedule_exists",
                    context={"path": str(output_path)},
                )
            return output_path
        return write_resolved_step_schedule(
            schedule,
            self.run_dir,
            overwrite=False,
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, _json_dumps(payload) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except ValueError as exc:
        raise ArtifactContractError(
            "artifact payload contains non-finite JSON values",
            code="artifact.non_finite_json",
            cause=exc,
        ) from exc


def _jsonl_dumps(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ArtifactContractError(
            "metric record contains non-finite JSON values",
            code="metric.non_finite_json",
            cause=exc,
        ) from exc


def _jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_json_dumps(dict(payload)))


def _json_file_matches(path: Path, expected_payload: Mapping[str, Any]) -> bool:
    try:
        actual_payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            "existing JSON artifact is not readable",
            code="artifact.existing_json_invalid",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    return actual_payload == dict(expected_payload)


def _metric_event_already_recorded(
    path: Path,
    record: Mapping[str, Any],
) -> bool:
    if not path.exists():
        return False
    target_identity = _metric_event_identity(record)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            existing = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(
                "existing metric stream record is not readable",
                code="metric.existing_stream_invalid",
                context={"path": str(path), "line_number": line_number},
                cause=exc,
            ) from exc
        if _metric_event_identity(existing) != target_identity:
            continue
        if existing == dict(record):
            return True
        raise ArtifactContractError(
            "metric stream already contains a conflicting event record",
            code="metric.event_conflict",
            context={
                "path": str(path),
                "line_number": line_number,
                "identity": dict(zip(_METRIC_IDENTITY_FIELDS, target_identity, strict=True)),
            },
        )
    return False


_METRIC_IDENTITY_FIELDS = (
    "event_type",
    "planned_step_id",
    "split",
    "name",
    "rank",
    "world_size",
    "reduction",
)


def _metric_event_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in _METRIC_IDENTITY_FIELDS)


def _yaml_file_matches(path: Path, expected_payload: Mapping[str, Any]) -> bool:
    try:
        actual_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ArtifactContractError(
            "existing YAML artifact is not readable",
            code="artifact.existing_yaml_invalid",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    return actual_payload == dict(expected_payload)


def _validate_name(value: str, *, field: str) -> None:
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ArtifactContractError(
            f"{field} must be a simple path component",
            code="artifact.name",
            context={"field": field, "value": value},
        )


def _validate_metric_split(split: str) -> None:
    if not split or "/" in split or "\\" in split or ".." in split:
        raise ArtifactContractError(
            "metric split must be a simple stream name",
            code="metric.split_path",
            context={"split": split},
        )


__all__ = ["RunArtifactManager"]
