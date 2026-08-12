"""Replicated-DDP checkpoint save choreography."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from src.artifacts.checkpoint_payload import (
    write_inference_checkpoint_payload_manifest,
)
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
        exact_training_state_callback: Callable[[Path], None] | None = None,
    ) -> CheckpointWriteResult:
        """Save inference state, optional exact state, then aliases in rank-safe order."""
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
                write_inference_checkpoint_payload_manifest(
                    staging_dir,
                    expected_base_model_path=base_model_path,
                    expected_base_config_sha256=base_config_sha256,
                    expected_tokenizer_sha256=tokenizer_sha256,
                )
                os.replace(staging_dir, checkpoint_dir)
                committed_this_call = True
                _fsync_directory(checkpoints_dir)

                if run_writer is not None and exact_training_state_callback is None:
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
            _raise_checkpoint_save_failed(status, step=step, rank_zero=True)

        if exact_training_state_callback is not None:
            callback_status = _run_exact_training_state_callback(
                accelerator=accelerator,
                checkpoint_dir=checkpoint_dir,
                callback=exact_training_state_callback,
            )
            if not callback_status["ok"]:
                _raise_checkpoint_save_failed(callback_status, step=step)

            alias_status: dict[str, Any] = {
                "ok": True,
                "error": None,
                "final_updated": False,
                "best_updated": False,
            }
            if is_main:
                try:
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
                    alias_status.update(
                        final_updated=final_updated,
                        best_updated=best_updated,
                    )
                except BaseException as exc:
                    rollback_errors: tuple[str, ...] = ()
                    try:
                        _restore_aliases(checkpoints_dir, alias_backups)
                    except BaseException as rollback_exc:
                        rollback_errors = (
                            f"alias_restore={_bounded_error(rollback_exc)}",
                        )
                    alias_status = {
                        "ok": False,
                        "error": _bounded_failure(exc, rollback_errors),
                        "final_updated": False,
                        "best_updated": False,
                    }
            alias_status = _broadcast_alias_status(accelerator, alias_status)
            if not alias_status["ok"]:
                _raise_checkpoint_save_failed(alias_status, step=step, rank_zero=True)
            if is_main:
                final_updated = alias_status["final_updated"]
                best_updated = alias_status["best_updated"]
        return CheckpointWriteResult(
            step=step,
            checkpoint_dir=checkpoint_dir,
            final_updated=final_updated if is_main else False,
            best_updated=best_updated if is_main else False,
        )


def _run_exact_training_state_callback(
    *,
    accelerator: Any,
    checkpoint_dir: Path,
    callback: Callable[[Path], None],
) -> dict[str, Any]:
    rank, world_size = _distributed_identity(accelerator)
    local_status: dict[str, Any] = {
        "ok": True,
        "rank": rank,
        "world_size": world_size,
        "error": None,
    }
    try:
        result = callback(checkpoint_dir)
        if result is not None:
            raise ArtifactContractError(
                "exact training-state callback must return None",
                code="checkpoint.exact_state_callback_result",
                context={"result_type": type(result).__name__},
            )
    except BaseException as exc:
        local_status.update(ok=False, error=_bounded_error(exc))

    try:
        reports = _gather_callback_statuses(
            accelerator,
            local_status,
            world_size=world_size,
        )
        validated = _validate_callback_statuses(reports, world_size=world_size)
    except BaseException as exc:
        return {
            "ok": False,
            "error": (
                "exact training-state callback status is invalid: "
                f"{_bounded_error(exc)}"
            )[:_MAX_COLLECTIVE_ERROR_CHARS],
        }

    failures = [report for report in validated if not report["ok"]]
    if not failures:
        return {"ok": True, "error": None}
    errors = "; ".join(
        f"rank {report['rank']}: {report['error']}" for report in failures
    )
    failed_ranks = [report["rank"] for report in failures]
    return {
        "ok": False,
        "error": (
            f"exact training-state callback failed on ranks {failed_ranks}: {errors}"
        )[:_MAX_COLLECTIVE_ERROR_CHARS],
    }


def _distributed_identity(accelerator: Any) -> tuple[int, int]:
    rank = getattr(accelerator, "process_index", 0)
    world_size = getattr(accelerator, "num_processes", 1)
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or rank < 0
        or rank >= world_size
    ):
        raise ArtifactContractError(
            "exact training-state callback has invalid distributed identity",
            code="checkpoint.exact_state_callback_identity",
            context={"rank": rank, "world_size": world_size},
        )
    return rank, world_size


def _gather_callback_statuses(
    accelerator: Any,
    local_status: Mapping[str, Any],
    *,
    world_size: int,
) -> Sequence[Any]:
    values = [dict(local_status)]
    if world_size == 1:
        return values
    gather = getattr(accelerator, "gather_object", None)
    if callable(gather):
        return gather(values)

    from accelerate.utils import gather_object

    return gather_object(values)


def _validate_callback_statuses(
    reports: Sequence[Any], *, world_size: int
) -> tuple[dict[str, Any], ...]:
    if isinstance(reports, (str, bytes)) or len(reports) != world_size:
        raise ArtifactContractError(
            "exact training-state callback rank inventory has the wrong size",
            code="checkpoint.exact_state_callback_status",
            context={"expected": world_size, "observed": len(reports)},
        )
    validated: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, Mapping):
            raise ArtifactContractError(
                "exact training-state callback returned a malformed rank status",
                code="checkpoint.exact_state_callback_status",
            )
        ok = report.get("ok")
        rank = report.get("rank")
        reported_world_size = report.get("world_size")
        error = report.get("error")
        if (
            not isinstance(ok, bool)
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(reported_world_size, bool)
            or not isinstance(reported_world_size, int)
            or reported_world_size != world_size
            or rank < 0
            or rank >= world_size
            or (ok and error is not None)
            or (not ok and (not isinstance(error, str) or not error))
        ):
            raise ArtifactContractError(
                "exact training-state callback returned a malformed rank status",
                code="checkpoint.exact_state_callback_status",
            )
        validated.append(
            {
                "ok": ok,
                "rank": rank,
                "world_size": reported_world_size,
                "error": error,
            }
        )
    validated.sort(key=lambda item: item["rank"])
    if [item["rank"] for item in validated] != list(range(world_size)):
        raise ArtifactContractError(
            "exact training-state callback rank inventory is incomplete or duplicated",
            code="checkpoint.exact_state_callback_status",
        )
    return tuple(validated)


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
        or not any(
            token in key for token in ("lora_A", "lora_B", "lora_magnitude_vector")
        )
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


def _broadcast_alias_status(accelerator: Any, status: dict[str, Any]) -> dict[str, Any]:
    received = _broadcast_status(accelerator, status)
    if not isinstance(received.get("final_updated"), bool) or not isinstance(
        received.get("best_updated"), bool
    ):
        raise ArtifactContractError(
            "checkpoint alias status collective returned an invalid descriptor",
            code="checkpoint.invalid_collective_status",
        )
    if not received["ok"] and (received["final_updated"] or received["best_updated"]):
        raise ArtifactContractError(
            "failed checkpoint alias status cannot report updated aliases",
            code="checkpoint.invalid_collective_status",
        )
    return received


def _raise_checkpoint_save_failed(
    status: Mapping[str, Any], *, step: int, rank_zero: bool = False
) -> None:
    owner = " on rank zero" if rank_zero else ""
    raise ArtifactContractError(
        f"checkpoint save failed{owner}: {status.get('error')}",
        code="checkpoint.save_failed",
        context={"step": step},
    )


def _capture_aliases(checkpoints_dir: Path) -> dict[str, bytes | None]:
    return {
        name: (path.read_bytes() if path.exists() else None)
        for name in ("final.json", "best.json")
        for path in (checkpoints_dir / name,)
    }


def _restore_aliases(
    checkpoints_dir: Path, backups: Mapping[str, bytes | None]
) -> None:
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
        if (
            payload.get("step") == step
            or payload.get("checkpoint_path") == expected_path
        ):
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
