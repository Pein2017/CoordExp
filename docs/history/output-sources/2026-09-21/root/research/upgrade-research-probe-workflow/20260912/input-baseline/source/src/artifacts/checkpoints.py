"""Replicated-DDP checkpoint save choreography."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from src.artifacts.run_writer import RunWriter
from src.common.errors import ArtifactContractError
from src.qwen.special_token_embeddings import (
    SpecialTokenEmbeddingInstallResult,
    save_special_token_embedding_deltas,
)


_MAX_COLLECTIVE_ERROR_CHARS = 1024
_ADAPTER_TENSOR_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = frozenset({_ADAPTER_TENSOR_FILE})


@dataclass(frozen=True)
class CheckpointWriteResult:
    step: int
    checkpoint_dir: Path
    final_updated: bool
    best_updated: bool


@dataclass(frozen=True)
class CheckpointWriter:
    """Save one canonical checkpoint; every replicated-DDP rank must call it."""

    run_dir: Path

    def write_checkpoint(
        self,
        *,
        step: int,
        accelerator: Any,
        model: Any,
        adapter_name: str,
        special_token_result: SpecialTokenEmbeddingInstallResult | None = None,
        base_model_path: Path | str | None = None,
        base_config_sha256: str | None = None,
        tokenizer_sha256: str | None = None,
        run_writer: RunWriter | None = None,
        is_final: bool = False,
        best_candidate: Mapping[str, Any] | None = None,
    ) -> CheckpointWriteResult:
        """Save a step atomically and synchronize one bounded outcome to all ranks."""
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ArtifactContractError(
                "checkpoint step must be a positive integer",
                code="checkpoint.invalid_step",
            )
        if not adapter_name.strip():
            raise ArtifactContractError(
                "checkpoint adapter name must be nonempty",
                code="checkpoint.invalid_adapter_name",
            )

        checkpoints_dir = self.run_dir / "checkpoints"
        checkpoint_dir = checkpoints_dir / f"step-{step}"
        staging_dir = checkpoints_dir / f".step-{step}.{uuid.uuid4().hex}.tmp"
        is_main = bool(accelerator.is_main_process)
        status: dict[str, Any] = {"ok": True, "error": None}
        final_updated = False
        best_updated = False

        accelerator.wait_for_everyone()
        if is_main:
            alias_backups = _capture_aliases(checkpoints_dir)
            committed_this_call = False
            try:
                if checkpoint_dir.exists():
                    raise ArtifactContractError(
                        "checkpoint step already exists",
                        code="checkpoint.step_exists",
                        context={"path": str(checkpoint_dir)},
                    )
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                adapter_dir = staging_dir / "adapter"
                adapter_dir.mkdir(parents=True)
                save_pretrained = getattr(unwrapped, "save_pretrained", None)
                if not callable(save_pretrained):
                    raise ArtifactContractError(
                        "adapter checkpoint requires save_pretrained",
                        code="checkpoint.adapter_unsavable",
                    )
                save_pretrained(
                    adapter_dir,
                    safe_serialization=True,
                    selected_adapters=[adapter_name],
                    save_embedding_layers=False,
                )
                _validate_adapter_payload(adapter_dir)
                if special_token_result is not None:
                    save_special_token_embedding_deltas(
                        special_token_result,
                        staging_dir / "special_token_embeddings",
                        base_model_path=base_model_path,
                        base_config_sha256=base_config_sha256,
                        tokenizer_sha256=tokenizer_sha256,
                    )
                os.replace(staging_dir, checkpoint_dir)
                committed_this_call = True
                _fsync_directory(checkpoints_dir)

                if run_writer is not None:
                    if is_final:
                        run_writer.write_final(step=step)
                        final_updated = True
                    if best_candidate is not None:
                        best_updated = _write_best_if_eligible(
                            run_writer=run_writer,
                            step=step,
                            candidate=best_candidate,
                        )
            except BaseException as exc:
                try:
                    rollback_errors = _rollback_after_failure(
                        staging_dir=staging_dir,
                        checkpoint_dir=checkpoint_dir,
                        checkpoints_dir=checkpoints_dir,
                        alias_backups=alias_backups,
                        committed_this_call=committed_this_call,
                        step=step,
                    )
                except BaseException as rollback_exc:
                    rollback_errors = (
                        f"rollback_internal={type(rollback_exc).__name__}",
                    )
                status = {
                    "ok": False,
                    "error": _bounded_failure(exc, rollback_errors),
                }

        status = _broadcast_status(accelerator, status)
        if not status.get("ok"):
            raise ArtifactContractError(
                f"checkpoint save failed on rank zero: {status.get('error')}",
                code="checkpoint.save_failed",
                context={"step": step},
            )
        return CheckpointWriteResult(
            step=step,
            checkpoint_dir=checkpoint_dir,
            final_updated=final_updated if is_main else False,
            best_updated=best_updated if is_main else False,
        )


def _validate_adapter_payload(adapter_dir: Path) -> None:
    tensor_path = adapter_dir / _ADAPTER_TENSOR_FILE
    if not (adapter_dir / "adapter_config.json").is_file() or not tensor_path.is_file():
        raise ArtifactContractError(
            "adapter checkpoint is missing its config or safetensor",
            code="checkpoint.adapter_payload_missing",
        )
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        empty = [key for key in keys if handle.get_tensor(key).numel() == 0]
    forbidden = [
        key
        for key in keys
        if "embed_tokens" in key
        or "word_embeddings" in key
        or "lm_head" in key
        or not any(token in key for token in ("lora_A", "lora_B", "lora_magnitude_vector"))
    ]
    required = {
        "lora_A": any("lora_A" in key for key in keys),
        "lora_B": any("lora_B" in key for key in keys),
        "lora_magnitude_vector": any("lora_magnitude_vector" in key for key in keys),
    }
    if forbidden:
        raise ArtifactContractError(
            "adapter checkpoint contains forbidden full-model tensors",
            code="checkpoint.adapter_forbidden_tensors",
            context={"keys": forbidden[:20]},
        )
    if empty or not all(required.values()):
        raise ArtifactContractError(
            "adapter checkpoint requires nonempty LoRA A/B and DoRA magnitude tensors",
            code="checkpoint.adapter_required_tensors_missing",
            context={"required": required, "empty_keys": empty[:20]},
        )


def _write_best_if_eligible(
    *, run_writer: RunWriter, step: int, candidate: Mapping[str, Any]
) -> bool:
    completed = candidate.get("completed") is True
    value = candidate.get("value")
    if not completed or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return run_writer.write_best(
        step=step,
        selector=str(candidate.get("selector", "eval/metric:max")),
        value=float(value),
        optimizer_update_status=str(candidate.get("optimizer_update_status", "")),
        finite_status=str(candidate.get("finite_status", "")),
        checkpoint_committed=True,
    )


def _broadcast_status(accelerator: Any, status: dict[str, Any]) -> dict[str, Any]:
    values = [status]
    broadcast = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcast):
        result = broadcast(values, from_process=0)
        if result is not None:
            values = result
    else:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(values, from_process=0)
    received = values[0]
    if not isinstance(received, Mapping) or not isinstance(received.get("ok"), bool):
        raise ArtifactContractError(
            "checkpoint status collective returned an invalid descriptor",
            code="checkpoint.invalid_collective_status",
        )
    return dict(received)


def _capture_aliases(checkpoints_dir: Path) -> dict[str, bytes | None]:
    return {
        name: (path.read_bytes() if path.exists() else None)
        for name in ("final.json", "best.json")
        for path in (checkpoints_dir / name,)
    }


def _restore_aliases(checkpoints_dir: Path, backups: Mapping[str, bytes | None]) -> None:
    for name, content in backups.items():
        path = checkpoints_dir / name
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def _rollback_after_failure(
    *,
    staging_dir: Path,
    checkpoint_dir: Path,
    checkpoints_dir: Path,
    alias_backups: Mapping[str, bytes | None],
    committed_this_call: bool,
    step: int,
) -> tuple[str, ...]:
    """Best-effort rollback that never prevents the status collective."""
    errors: list[str] = []
    try:
        shutil.rmtree(staging_dir, ignore_errors=True)
    except BaseException as exc:
        errors.append(f"staging_cleanup={_bounded_error(exc)}")
    try:
        _restore_aliases(checkpoints_dir, alias_backups)
    except BaseException as exc:
        errors.append(f"alias_restore={_bounded_error(exc)}")

    # If rollback left an alias selecting this step, retaining the fully committed
    # payload is safer than deleting it and publishing a dangling selector.
    alias_may_select_step = False
    try:
        alias_may_select_step = _alias_selects_step(checkpoints_dir, step=step)
    except BaseException as exc:
        alias_may_select_step = True
        errors.append(f"alias_safety_check={_bounded_error(exc)}")
    if committed_this_call and not alias_may_select_step:
        try:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        except BaseException as exc:
            errors.append(f"checkpoint_cleanup={_bounded_error(exc)}")
    elif committed_this_call and alias_may_select_step:
        errors.append("checkpoint_retained=alias_may_select_failed_step")
    return tuple(errors)


def _alias_selects_step(checkpoints_dir: Path, *, step: int) -> bool:
    expected_path = f"checkpoints/step-{step}"
    for name in ("final.json", "best.json"):
        path = checkpoints_dir / name
        if not path.exists():
            continue
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except BaseException:
            return True
        if payload.get("step") == step or payload.get("checkpoint_path") == expected_path:
            return True
    return False


def _bounded_failure(exc: BaseException, rollback_errors: tuple[str, ...]) -> str:
    primary = _bounded_error(exc)
    if not rollback_errors:
        return primary
    return f"{primary}; rollback: {'; '.join(rollback_errors)}"[
        :_MAX_COLLECTIVE_ERROR_CHARS
    ]


def _bounded_error(exc: BaseException) -> str:
    if isinstance(exc, ArtifactContractError):
        value = f"{exc.code}: {exc}"
    else:
        value = f"{type(exc).__name__}: {exc}"
    return value[:_MAX_COLLECTIVE_ERROR_CHARS]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
