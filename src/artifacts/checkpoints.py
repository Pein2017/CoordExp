"""Checkpoint metadata and alias writer."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.adapters.dora import DoraAdapterSetupReceipt
from src.artifacts.manager import RunArtifactManager
from src.artifacts.metric_stream import MetricStreamEvent
from src.common.errors import ArtifactContractError
from src.optim.trainable_surface import TrainableSurfaceReceipt
from src.qwen.special_token_embeddings import (
    SPECIAL_TOKEN_EMBEDDINGS_JSON,
    SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS,
    SpecialTokenEmbeddingInstallResult,
    save_special_token_embedding_deltas,
)


BEST_ACC_TOP1_SELECTOR = "eval.forward/acc_top1:max"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = frozenset({"adapter_model.safetensors", "adapter_model.bin"})
FORBIDDEN_ADAPTER_PAYLOAD_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
)


@dataclass(frozen=True)
class CheckpointWriteResult:
    checkpoint_id: str
    checkpoint_dir: Path
    metadata_path: Path
    final_alias_path: Path | None
    best_alias_path: Path | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointWriter:
    manager: RunArtifactManager

    def write_checkpoint(
        self,
        *,
        planned_step_id: int,
        model: Any | None,
        adapter_receipt: DoraAdapterSetupReceipt | Mapping[str, Any] | None,
        special_token_result: SpecialTokenEmbeddingInstallResult | None,
        trainable_surface: TrainableSurfaceReceipt | Mapping[str, Any],
        processor_identity: Mapping[str, Any],
        resolved_config_fingerprint: str,
        schedule_identity: Mapping[str, Any],
        metric_status: Mapping[str, Any],
        optimizer_update_status: str,
        trigger_reasons: Sequence[str],
        is_final: bool = False,
        best_metric_event: MetricStreamEvent | Mapping[str, Any] | None = None,
        base_model_path: Path | str | None = None,
        base_config_sha256: str | None = None,
        tokenizer_sha256: str | None = None,
    ) -> CheckpointWriteResult:
        if planned_step_id <= 0:
            raise ArtifactContractError(
                "checkpoint planned_step_id must be positive",
                code="checkpoint.planned_step_id",
                context={"planned_step_id": planned_step_id},
            )
        best_record = _metric_record(best_metric_event)
        checkpoint_best_rejection_reason = _checkpoint_best_rejection_reason(
            optimizer_update_status=optimizer_update_status,
            metric_status=metric_status,
        )
        _validate_best_checkpoint_status(
            record=best_record,
            optimizer_update_status=optimizer_update_status,
            metric_status=metric_status,
            checkpoint_rejection_reason=checkpoint_best_rejection_reason,
        )
        checkpoint_id = f"step-{planned_step_id}"
        checkpoint_dir = self.manager.run_dir / "checkpoints" / checkpoint_id
        metadata_path = checkpoint_dir / "checkpoint.json"
        existing_metadata = _read_json_if_exists(metadata_path)

        adapter_payload = self._save_adapter_payload(
            model=model,
            adapter_receipt=adapter_receipt,
            checkpoint_dir=checkpoint_dir,
            reuse_existing=existing_metadata is not None,
        )
        special_token_payload = self._save_special_token_payload(
            special_token_result=special_token_result,
            checkpoint_dir=checkpoint_dir,
            base_model_path=base_model_path,
            base_config_sha256=base_config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            reuse_existing=existing_metadata is not None,
        )
        manifest = self.manager.read_manifest()
        best_selection = _best_selection_decision(
            manifest,
            best_record,
            checkpoint_rejection_reason=checkpoint_best_rejection_reason,
        )
        metadata = {
            "checkpoint_id": checkpoint_id,
            "planned_step_id": planned_step_id,
            "checkpoint_path": self.manager.relative_artifact_path(checkpoint_dir),
            "adapter": adapter_payload,
            "special_token_embeddings": special_token_payload,
            "processor_identity": dict(processor_identity),
            "resolved_config_fingerprint": resolved_config_fingerprint,
            "schedule_identity": dict(schedule_identity),
            "metric_status": dict(metric_status),
            "trainable_surface": _artifact_dict(trainable_surface),
            "optimizer_update_status": optimizer_update_status,
            "trigger_reasons": list(trigger_reasons),
            "best_selection": best_selection,
            "resume_state": {
                "optimizer": "not_saved_v1",
                "scheduler": "not_saved_v1",
                "scaler": "not_saved_v1",
                "dataloader": "not_saved_v1",
                "iterator": "not_saved_v1",
                "rng": "not_saved_v1",
            },
        }
        _write_json_or_reuse(metadata_path, metadata, code="checkpoint.exists")

        final_alias_path = None
        if is_final:
            final_alias_path = self.manager.run_dir / "checkpoints" / "checkpoint-final.json"
            _write_json_or_reuse(
                final_alias_path,
                _alias_payload(
                    alias="final",
                    checkpoint_id=checkpoint_id,
                    planned_step_id=planned_step_id,
                    metadata_path=self.manager.relative_artifact_path(metadata_path),
                ),
                code="checkpoint.alias_exists",
            )

        best_alias_path = None
        best_acc_top1 = None
        if best_selection["selected"]:
            best_alias_path = self.manager.run_dir / "checkpoints" / "best_acc_top1.json"
            best_acc_top1 = _alias_payload(
                alias="best_acc_top1",
                checkpoint_id=checkpoint_id,
                planned_step_id=planned_step_id,
                metadata_path=self.manager.relative_artifact_path(metadata_path),
                metric=best_record,
                selector=BEST_ACC_TOP1_SELECTOR,
            )
            _write_json(best_alias_path, best_acc_top1)

        self.manager.register_checkpoint(
            checkpoint_id=checkpoint_id,
            planned_step_id=planned_step_id,
            metadata_path=metadata_path,
            final_alias_path=final_alias_path,
            best_alias_path=best_alias_path,
            best_acc_top1=best_acc_top1,
        )
        return CheckpointWriteResult(
            checkpoint_id=checkpoint_id,
            checkpoint_dir=checkpoint_dir,
            metadata_path=metadata_path,
            final_alias_path=final_alias_path,
            best_alias_path=best_alias_path,
            metadata=metadata,
        )

    def _save_adapter_payload(
        self,
        *,
        model: Any | None,
        adapter_receipt: DoraAdapterSetupReceipt | Mapping[str, Any] | None,
        checkpoint_dir: Path,
        reuse_existing: bool,
    ) -> dict[str, Any]:
        if adapter_receipt is None:
            return {"enabled": False}
        adapter_dir = checkpoint_dir / "adapter"
        if reuse_existing:
            if not adapter_dir.exists():
                raise ArtifactContractError(
                    "checkpoint metadata exists but adapter payload is missing",
                    code="checkpoint.adapter_payload_missing",
                    context={"path": str(adapter_dir)},
                )
        else:
            save_pretrained = getattr(model, "save_pretrained", None)
            if model is None or not callable(save_pretrained):
                raise ArtifactContractError(
                    "adapter checkpoint requires a model with save_pretrained",
                    code="checkpoint.adapter_model_unsavable",
                )
            self._save_adapter_to_staging(
                save_pretrained=save_pretrained,
                adapter_dir=adapter_dir,
            )
        files = _relative_file_list(adapter_dir, root=self.manager.run_dir)
        _validate_adapter_payload(files, adapter_dir=adapter_dir)
        return {
            "enabled": True,
            "payload_path": self.manager.relative_artifact_path(adapter_dir),
            "files": files,
            "receipt": _artifact_dict(adapter_receipt),
        }

    def _save_adapter_to_staging(
        self,
        *,
        save_pretrained: Any,
        adapter_dir: Path,
    ) -> None:
        if adapter_dir.exists():
            raise ArtifactContractError(
                "adapter checkpoint payload already exists before save",
                code="checkpoint.adapter_payload_exists",
                context={"path": str(adapter_dir)},
            )
        staging_dir = adapter_dir.with_name(f".{adapter_dir.name}.{os.getpid()}.tmp")
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        try:
            save_pretrained(staging_dir)
            staged_files = _relative_file_list(staging_dir, root=staging_dir)
            _validate_adapter_payload(staged_files, adapter_dir=staging_dir)
            adapter_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_dir, adapter_dir)
            _fsync_directory(adapter_dir.parent)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def _save_special_token_payload(
        self,
        *,
        special_token_result: SpecialTokenEmbeddingInstallResult | None,
        checkpoint_dir: Path,
        base_model_path: Path | str | None,
        base_config_sha256: str | None,
        tokenizer_sha256: str | None,
        reuse_existing: bool,
    ) -> dict[str, Any]:
        if special_token_result is None:
            return {"enabled": False}
        _require_special_token_identity_sha(
            base_config_sha256,
            field="base_config_sha256",
            checkpoint_dir=checkpoint_dir,
        )
        _require_special_token_identity_sha(
            tokenizer_sha256,
            field="tokenizer_sha256",
            checkpoint_dir=checkpoint_dir,
        )
        output_dir = checkpoint_dir / "special_token_embeddings"
        if reuse_existing:
            return _special_token_payload_from_existing(
                output_dir,
                manager=self.manager,
                install_receipt=special_token_result.receipt.to_artifact_dict(),
            )
        payload = save_special_token_embedding_deltas(
            special_token_result,
            output_dir,
            base_model_path=base_model_path,
            base_config_sha256=base_config_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )
        artifact = payload.to_artifact_dict()
        artifact["tensor_path"] = self.manager.relative_artifact_path(payload.tensor_path)
        artifact["metadata_path"] = self.manager.relative_artifact_path(payload.metadata_path)
        artifact["enabled"] = True
        artifact["install_receipt"] = special_token_result.receipt.to_artifact_dict()
        return artifact


def _require_special_token_identity_sha(
    value: str | None,
    *,
    field: str,
    checkpoint_dir: Path,
) -> None:
    if value is None or not str(value).strip():
        raise ArtifactContractError(
            "checkpoint special-token embedding payload requires runtime identity SHA evidence",
            code="checkpoint.special_token_identity_missing",
            context={
                "missing_field": field,
                "checkpoint_dir": str(checkpoint_dir),
            },
        )


def _artifact_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_artifact_dict"):
        return value.to_artifact_dict()
    return dict(value)


def _metric_record(
    event: MetricStreamEvent | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if event is None:
        return None
    if isinstance(event, MetricStreamEvent):
        return event.to_record()
    return dict(event)


def _validate_best_checkpoint_status(
    *,
    record: Mapping[str, Any] | None,
    optimizer_update_status: str,
    metric_status: Mapping[str, Any],
    checkpoint_rejection_reason: str | None,
) -> None:
    if record is None or not _is_best_selector_record(record):
        return
    missing_status_fields = [
        field
        for field in ("finite_status", "warning_status")
        if field not in metric_status
    ]
    if missing_status_fields:
        raise ArtifactContractError(
            "best checkpoint selection requires explicit checkpoint status",
            code="checkpoint.best_selector_status_missing",
            context={
                "missing_fields": missing_status_fields,
                "optimizer_update_status": optimizer_update_status,
            },
        )
    mismatch_fields = _record_checkpoint_status_mismatches(
        record=record,
        optimizer_update_status=optimizer_update_status,
        metric_status=metric_status,
    )
    if not mismatch_fields and (
        checkpoint_rejection_reason is None
        or record.get("selector_eligible") is not True
    ):
        return
    raise ArtifactContractError(
        "best checkpoint metric status conflicts with checkpoint status",
        code="checkpoint.best_selector_status_mismatch",
        context={
            "checkpoint_rejection_reason": checkpoint_rejection_reason,
            "mismatch_fields": mismatch_fields,
            "record_optimizer_update_status": record.get("optimizer_update_status"),
            "checkpoint_optimizer_update_status": optimizer_update_status,
            "record_finite_status": record.get("finite_status"),
            "checkpoint_finite_status": metric_status.get("finite_status"),
            "record_warning_status": record.get("warning_status"),
            "checkpoint_warning_status": metric_status.get("warning_status"),
        },
    )


def _record_checkpoint_status_mismatches(
    *,
    record: Mapping[str, Any],
    optimizer_update_status: str,
    metric_status: Mapping[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    if record.get("optimizer_update_status") != optimizer_update_status:
        mismatches.append("optimizer_update_status")
    finite_status = metric_status.get("finite_status")
    if finite_status is not None and record.get("finite_status") != finite_status:
        mismatches.append("finite_status")
    warning_status = metric_status.get("warning_status")
    if warning_status is not None and record.get("warning_status") != warning_status:
        mismatches.append("warning_status")
    return mismatches


def _checkpoint_best_rejection_reason(
    *,
    optimizer_update_status: str,
    metric_status: Mapping[str, Any],
) -> str | None:
    if optimizer_update_status != "applied":
        return "checkpoint_update_not_applied"
    finite_status = metric_status.get("finite_status")
    if finite_status is not None and finite_status != "finite":
        return "checkpoint_non_finite"
    return None


def _best_selection_decision(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    *,
    checkpoint_rejection_reason: str | None,
) -> dict[str, Any]:
    if record is None:
        return {
            "selector": BEST_ACC_TOP1_SELECTOR,
            "candidate_seen": False,
            "selected": False,
            "reason": "no_candidate",
        }
    reason = checkpoint_rejection_reason or _best_rejection_reason(record)
    if reason is not None:
        return {
            "selector": BEST_ACC_TOP1_SELECTOR,
            "candidate_seen": True,
            "selected": False,
            "reason": reason,
            "metric": dict(record),
        }
    current = manifest.get("checkpoints", {}).get("best_acc_top1")
    if current:
        current_value = current.get("metric", {}).get("value")
        if current_value is not None and record["value"] <= current_value:
            return {
                "selector": BEST_ACC_TOP1_SELECTOR,
                "candidate_seen": True,
                "selected": False,
                "reason": "not_improved",
                "metric": dict(record),
                "current_best": dict(current),
            }
    return {
        "selector": BEST_ACC_TOP1_SELECTOR,
        "candidate_seen": True,
        "selected": True,
        "reason": "improved",
        "metric": dict(record),
    }


def _best_rejection_reason(record: Mapping[str, Any]) -> str | None:
    if not _is_best_selector_record(record):
        return "selector_mismatch"
    if record.get("value") is None:
        return "metric_unavailable"
    if record.get("reduction") == "rank_local":
        return "metric_rank_local"
    if record.get("optimizer_update_status") != "applied":
        return "metric_update_not_applied"
    if record.get("finite_status") != "finite":
        return "metric_non_finite"
    return None


def _is_best_selector_record(record: Mapping[str, Any]) -> bool:
    return record.get("split") == "eval.forward" and record.get("name") == "acc_top1"


def _alias_payload(
    *,
    alias: str,
    checkpoint_id: str,
    planned_step_id: int,
    metadata_path: str,
    metric: Mapping[str, Any] | None = None,
    selector: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "alias": alias,
        "checkpoint_id": checkpoint_id,
        "planned_step_id": planned_step_id,
        "metadata_path": metadata_path,
    }
    if selector is not None:
        payload["selector"] = selector
    if metric is not None:
        payload["metric"] = dict(metric)
    return payload


def _relative_file_list(path: Path, *, root: Path) -> list[str]:
    return sorted(
        file_path.resolve().relative_to(root).as_posix()
        for file_path in path.rglob("*")
        if file_path.is_file()
    )


def _validate_adapter_payload(files: Sequence[str], *, adapter_dir: Path) -> None:
    if not files:
        raise ArtifactContractError(
            "adapter save_pretrained produced no files",
            code="checkpoint.adapter_payload_empty",
            context={"path": str(adapter_dir)},
        )
    basenames = {Path(path).name for path in files}
    for basename in sorted(basenames):
        if basename in FORBIDDEN_ADAPTER_PAYLOAD_NAMES or (
            basename.startswith("model-") and basename.endswith(".safetensors")
        ):
            raise ArtifactContractError(
                "adapter payload contains full-model weight artifact",
                code="checkpoint.adapter_payload_forbidden_file",
                context={"path": str(adapter_dir), "file": basename},
            )
    if ADAPTER_CONFIG_NAME not in basenames:
        raise ArtifactContractError(
            "adapter payload is missing adapter_config.json",
            code="checkpoint.adapter_payload_missing_config",
            context={"path": str(adapter_dir), "files": list(files)},
        )
    if not basenames.intersection(ADAPTER_WEIGHT_NAMES):
        raise ArtifactContractError(
            "adapter payload is missing adapter weight file",
            code="checkpoint.adapter_payload_missing_weights",
            context={"path": str(adapter_dir), "files": list(files)},
        )


def _special_token_payload_from_existing(
    output_dir: Path,
    *,
    manager: RunArtifactManager,
    install_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    tensor_path = output_dir / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    metadata_path = output_dir / SPECIAL_TOKEN_EMBEDDINGS_JSON
    if not tensor_path.exists() or not metadata_path.exists():
        raise ArtifactContractError(
            "checkpoint metadata exists but special-token payload is incomplete",
            code="checkpoint.special_token_payload_missing",
            context={
                "tensor_path": str(tensor_path),
                "metadata_path": str(metadata_path),
            },
        )
    metadata = _read_json(metadata_path)
    return {
        "enabled": True,
        "tensor_path": manager.relative_artifact_path(tensor_path),
        "metadata_path": manager.relative_artifact_path(metadata_path),
        "tensor_key": metadata["tensor_key"],
        "tensor_shape": list(metadata["tensor_shape"]),
        "tensor_dtype": metadata["tensor_dtype"],
        "metadata": metadata,
        "install_receipt": dict(install_receipt),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: (_raise_invalid_json_constant(path, constant)),
        )
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            "existing checkpoint JSON artifact is not readable",
            code="checkpoint.existing_json_invalid",
            context={"path": str(path)},
            cause=exc,
        ) from exc


def _raise_invalid_json_constant(path: Path, constant: str) -> None:
    raise ArtifactContractError(
        "existing checkpoint JSON artifact contains non-finite values",
        code="checkpoint.existing_json_non_finite",
        context={"path": str(path), "constant": constant},
    )


def _write_json_or_reuse(
    path: Path,
    payload: Mapping[str, Any],
    *,
    code: str,
) -> None:
    existing = _read_json_if_exists(path)
    if existing is not None:
        if existing != dict(payload):
            raise ArtifactContractError(
                "checkpoint artifact already exists with different content",
                code=code,
                context={"path": str(path)},
            )
        return
    _write_json(path, payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(_json_dumps(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


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
            "checkpoint payload contains non-finite JSON values",
            code="checkpoint.non_finite_json",
            cause=exc,
        ) from exc


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = ["CheckpointWriteResult", "CheckpointWriter"]
