"""Training pipeline assembly for the V1 supervised smoke."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import struct
from typing import Any
import zlib

import torch

try:
    from accelerate import Accelerator
    from accelerate.utils import broadcast_object_list
except ImportError:  # pragma: no cover - exercised only in stripped environments.
    Accelerator = None  # type: ignore[assignment]
    broadcast_object_list = None  # type: ignore[assignment]

from src.adapters import (
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
    setup_dora_adapter,
)
from src.augmentation.factory import build_augmentation_processor
from src.augmentation.processor import AugmentationMaterializationResult
from src.artifacts import CheckpointWriter, RunWriter
from src.common.errors import RuntimeContractError
from src.config.loader import load_train_config
from src.config.models import RunDirectory
from src.config.paths import resolve_run_directory
from src.config.resolve import resolve_qwen_runtime_controls
from src.data import load_raw_examples
from src.eval import ForwardEvalRunner
from src.losses import LossRunner, build_token_vocabulary_groups
from src.optim import (
    build_optimizer_and_scheduler,
    build_optimizer_group_plan,
    build_scheduler_plan,
    build_trainable_surface_receipt,
)
from src.packing import PackedSequence, build_packed_supervision, plan_packed_sequences
from src.qwen import (
    QwenImageEncoding,
    attach_qwen_image_processor,
    build_default_special_token_selection,
    build_qwen_position_inputs,
    encode_rendered_example,
    load_qwen_components,
)
from src.qwen.special_token_embeddings import (
    load_default_special_token_embedding_source_gate_evidence,
    install_special_token_embedding_deltas,
    load_special_token_embedding_deltas,
)
from src.runtime import (
    TrainRuntime,
    seed_training_runtime,
    validate_accelerator_runtime,
)
from src.supervision import (
    build_token_sequence_from_packed_supervision,
    index_token_atoms_by_pack,
)
from src.templates import render_example
from src.training.schedule import ResolvedStepSchedule, resolve_planned_step_schedule
from src.training.pack_cache import (
    DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS,
    PACKING_CACHE_MATERIALIZATION_STRATEGY,
    PackingCacheInvalidError,
    build_packing_cache_materialization,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    cache_dir_for_fingerprint,
    load_all_micro_steps_from_cache,
    load_cache_manifest,
    load_rank_micro_steps_from_cache,
    manifest_path,
    write_micro_step_cache,
)
from src.training.supervised_trainer import (
    CompletedStepObservation,
    SupervisedMicroStep,
    SupervisedTrainer,
)


TRAIN_SPLIT = "train"
_PACK_CACHE_WORKER_CONTEXT: dict[str, Any] | None = None
BEST_EVAL_SELECTOR_NAME = "acc_top1"

_RANK_REPORT_MAGIC = b"CRG1"
_RANK_REPORT_HEADER = struct.Struct("!4sQQIIIIQQI")
_RANK_REPORT_MAX_PAYLOAD_BYTES = 64 * 1024
_RANK_REPORT_FRAME_BYTES = _RANK_REPORT_HEADER.size + _RANK_REPORT_MAX_PAYLOAD_BYTES
_RANK_REPORT_CONTROL_TIMEOUT_SECONDS = 120


def build_repeating_micro_step_stream(
    base_micro_steps: Sequence[SupervisedMicroStep],
    schedule: ResolvedStepSchedule,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[SupervisedMicroStep]:
    if not base_micro_steps:
        raise ValueError("base_micro_steps must contain at least one micro-step")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be inside world_size")
    if schedule.runtime_batch.world_size != world_size:
        raise ValueError("stream world_size must match schedule runtime_batch")
    total_rank_local_micro_steps = (
        schedule.resolved_max_steps
        * schedule.runtime_batch.resolved_grad_accum_steps
    )

    def iter_repeated() -> Iterator[SupervisedMicroStep]:
        for index in range(total_rank_local_micro_steps):
            planned_step_index = (
                index // schedule.runtime_batch.resolved_grad_accum_steps
            )
            local_accum_index = (
                index % schedule.runtime_batch.resolved_grad_accum_steps
            )
            global_micro_step_index = (
                planned_step_index * schedule.runtime_batch.effective_batch_size
                + local_accum_index * world_size
                + rank
            )
            yield base_micro_steps[global_micro_step_index % len(base_micro_steps)]

    return iter_repeated()


def _scheduler_lr_metrics(scheduler_artifact: Any) -> dict[str, float]:
    if not isinstance(scheduler_artifact, Mapping):
        return {}
    learning_rates = scheduler_artifact.get("learning_rates")
    if not isinstance(learning_rates, Sequence):
        return {}
    metrics: dict[str, float] = {}
    for item in learning_rates:
        if not isinstance(item, Mapping):
            continue
        group_index = item.get("group_index")
        lr = item.get("lr")
        if group_index is None or lr is None:
            continue
        metrics[f"lr/group_{int(group_index)}"] = float(lr)
    return metrics


def _initialize_artifact_owner(
    *,
    accelerator: Any,
    run_directory: RunDirectory,
    run_id: str,
    created_at: str,
    resolved_config: Any,
) -> RunWriter | None:
    writer: RunWriter | None = None
    status: dict[str, Any] | None = None
    if bool(accelerator.is_main_process):
        try:
            writer = RunWriter.initialize(
                run_dir=run_directory.run_dir,
                run_id=run_id,
                run_name=run_directory.run_name,
                artifact_root=run_directory.artifact_root,
                collision_outcome=run_directory.collision_policy,
                created_at=created_at,
                config_fingerprint=resolved_config.fingerprint,
                resolved_config=resolved_config.to_artifact_dict(),
                world_size=int(accelerator.num_processes),
            )
            status = {"ok": True}
        except Exception as exc:
            status = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    objects: list[Any] = [status]
    if int(accelerator.num_processes) > 1:
        if broadcast_object_list is None:
            raise RuntimeContractError(
                "artifact initialization handshake requires accelerate",
                code="runtime.artifact_init_broadcast_unavailable",
            )
        broadcast_object_list(objects, from_process=0)
    shared = objects[0]
    if not isinstance(shared, Mapping) or not bool(shared.get("ok")):
        raise RuntimeContractError(
            "rank zero failed to initialize shared run artifacts",
            code="runtime.artifact_initialization_failed",
            context={
                "error_type": str(
                    shared.get("error_type", "unknown")
                    if isinstance(shared, Mapping)
                    else "invalid_status"
                ),
                "error": str(
                    shared.get("error", "unknown error")
                    if isinstance(shared, Mapping)
                    else shared
                ),
            },
        )
    if bool(accelerator.is_main_process):
        return writer
    return None


def _train_logging_handler(
    writer: RunWriter | None, lifecycle: dict[str, Any], runtime: Any
) -> Any:

    def handle(observation: CompletedStepObservation) -> None:
        lifecycle.update(
            completed_steps=observation.planned_step_id,
            consumed_packs=int(lifecycle.get("consumed_packs", 0))
            + observation.micro_step_count,
            optimizer_update_status=observation.optimizer_update_status,
            finite_status=observation.finite_status,
        )
        loss_bundle = dict(observation.loss_bundle_artifact)
        metrics = loss_bundle.get("metrics", {})
        scalar_metrics = _scheduler_lr_metrics(observation.scheduler_artifact)
        if isinstance(metrics, Mapping):
            scalar_metrics.update({str(name): value for name, value in metrics.items()})
        gathered = runtime.gather_metrics(
            scalar_metrics,
            planned_step_id=observation.planned_step_id,
            split=TRAIN_SPLIT,
        )
        reduced = gathered.get("metrics") if isinstance(gathered, Mapping) else None
        if not isinstance(reduced, Mapping):
            raise RuntimeContractError(
                "train metric reduction returned no scalar mapping",
                code="runtime.train_metric_reduction_failed",
            )
        row: dict[str, Any] = {
            "step": observation.planned_step_id,
            "split": TRAIN_SPLIT,
            "micro_step_count": observation.micro_step_count,
            "optimizer_update_status": observation.optimizer_update_status,
            "finite_status": observation.finite_status,
            **dict(reduced),
        }
        _append_logging_row_shared(writer=writer, row=row, runtime=runtime)

    return handle


def _append_logging_row_shared(
    *, writer: RunWriter | None, row: Mapping[str, Any], runtime: Any
) -> None:
    """Append on rank zero and make its bounded outcome common to every rank."""
    accelerator = getattr(runtime, "accelerator", runtime)
    is_main = bool(
        getattr(runtime, "is_main_process", getattr(accelerator, "is_main_process", True))
    )
    status: dict[str, Any] = {"ok": True}
    if is_main:
        try:
            if writer is None:
                raise RuntimeError("rank zero has no run writer")
            writer.append_logging_row(row)
        except BaseException as exc:
            status = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:1024],
            }
    values: list[Any] = [status]
    if int(
        getattr(runtime, "world_size", getattr(accelerator, "num_processes", 1))
    ) > 1:
        broadcast = getattr(accelerator, "broadcast_object_list", None)
        if callable(broadcast):
            result = broadcast(values, from_process=0)
            if result is not None:
                values = result
        elif broadcast_object_list is not None:
            broadcast_object_list(values, from_process=0)
        else:
            raise RuntimeContractError(
                "logging outcome broadcast requires accelerate",
                code="runtime.logging_broadcast_unavailable",
            )
    shared = values[0]
    if not isinstance(shared, Mapping) or not bool(shared.get("ok")):
        error = shared.get("error", "invalid status") if isinstance(shared, Mapping) else shared
        raise RuntimeContractError(
            f"rank zero logging append failed: {error}",
            code="runtime.logging_append_failed",
        )


def run_training_pipeline(config_path: str | Path) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    resolved_config = load_train_config(config_path)
    config = resolved_config.config
    accelerator = _build_accelerator(config.training.precision)
    validate_accelerator_runtime(
        accelerator,
        expected_mixed_precision=config.training.precision,
    )
    run_directory = _resolve_shared_run_directory(
        config,
        cwd=repo_root,
        accelerator=accelerator,
    )
    created_at = datetime.now(UTC).isoformat()
    run_id = _run_id(config.run.name, resolved_config.fingerprint)
    writer = _initialize_artifact_owner(
        accelerator=accelerator,
        run_directory=run_directory,
        run_id=run_id,
        created_at=created_at,
        resolved_config=resolved_config,
    )
    lifecycle: dict[str, Any] = {
        "completed_steps": 0,
        "consumed_packs": 0,
        "checkpoint_event_count": 0,
        "optimizer_update_status": None,
        "finite_status": None,
    }
    try:
        return _run_initialized_training(
            repo_root=repo_root,
            resolved_config=resolved_config,
            config=config,
            accelerator=accelerator,
            run_directory=run_directory,
            run_id=run_id,
            writer=writer,
            lifecycle=lifecycle,
        )
    except BaseException as exc:
        if writer is not None:
            try:
                writer.finalize(
                    status="failed",
                    updated_at=datetime.now(UTC).isoformat(),
                    completed_steps=int(lifecycle["completed_steps"]),
                    consumed_packs=int(lifecycle["consumed_packs"]),
                    checkpoint_event_count=int(lifecycle["checkpoint_event_count"]),
                    optimizer_update_status=lifecycle["optimizer_update_status"],
                    finite_status=lifecycle["finite_status"],
                    terminal_error=f"{type(exc).__name__}: {exc}",
                )
            except BaseException:
                pass
        raise


def prepare_training_pack_caches(config_path: str | Path) -> dict[str, Any]:
    """Materialize all packing caches before distributed model startup."""

    repo_root = Path.cwd().resolve()
    resolved_config = load_train_config(config_path)
    config = resolved_config.config
    seed_training_runtime(
        config.runtime.seed,
        deterministic=False,
        phase="pack_cache_preparation",
    )
    components = load_qwen_components(config, load_model=False)
    vocab_groups = build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )
    train_cache = _resolve_or_build_train_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=None,
        rank=0,
    )
    eval_cache = _resolve_eval_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=None,
        rank=0,
    )
    return {
        "entry_config_path": str(resolved_config.entry_config_path),
        "resolved_config_fingerprint": resolved_config.fingerprint,
        "model_loaded": False,
        "train": _pack_cache_preparation_receipt(train_cache),
        "eval": (
            None
            if eval_cache is None
            else _pack_cache_preparation_receipt(eval_cache)
        ),
    }


def _run_initialized_training(
    *,
    repo_root: Path,
    resolved_config: Any,
    config: Any,
    accelerator: Any,
    run_directory: RunDirectory,
    run_id: str,
    writer: RunWriter | None,
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    seed_training_runtime(
        config.runtime.seed,
        deterministic=False,
        phase="pipeline_assembly",
    )
    components = load_qwen_components(config, load_model=True)
    if components.model is None:
        raise RuntimeContractError(
            "training pipeline requires load_qwen_components(..., load_model=True)",
            code="training.model_not_loaded",
            context={"base_model": config.model.base_model},
        )
    runtime_controls = resolve_qwen_runtime_controls(
        config,
        tokenizer_vocab_size=components.token_identity.tokenizer_vocab_size,
        model_logits_dtype=config.training.precision,
    )
    del runtime_controls

    adapter_evidence = load_default_adapter_source_gate_evidence(repo_root)
    adapter_plan = build_adapter_setup_plan(
        config.adapter,
        adapter_evidence,
        base_model_path=components.base_model_path,
    )
    adapter_plan_mode = getattr(adapter_plan, "mode", None)
    adapter_result = setup_dora_adapter(components.model, adapter_plan)
    model = adapter_result.model

    special_token_selection = build_default_special_token_selection(
        config.model.special_token_embeddings,
        components.token_identity,
    )
    special_token_evidence = load_default_special_token_embedding_source_gate_evidence(
        repo_root
    )
    special_token_result = install_special_token_embedding_deltas(
        model,
        special_token_selection,
        source_gate=special_token_evidence,
    )
    model = special_token_result.model
    if adapter_plan_mode == "warm_start_expand_dora":
        if adapter_plan.repaired_embedding_payload_path is None:
            raise RuntimeContractError(
                "warm_start_expand_dora requires repaired embedding payload path",
                code="adapter.warm_start_embedding_payload_required",
            )
        load_special_token_embedding_deltas(
            special_token_result,
            adapter_plan.repaired_embedding_payload_path,
            expected_base_model_path=components.base_model_path,
            expected_base_config_sha256=components.base_config_sha256,
            expected_tokenizer_sha256=components.tokenizer_sha256,
        )
    enable_training_memory_savers(model)

    vocab_groups = build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )
    train_cache = _resolve_or_build_train_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=accelerator,
        rank=int(accelerator.process_index),
    )
    schedule = resolve_planned_step_schedule(
        config,
        packs_per_epoch=train_cache["micro_step_count"],
        world_size=int(accelerator.num_processes),
        source_config_path=str(resolved_config.entry_config_path),
    )
    if writer is not None:
        writer.bind_schedule(resolved_max_steps=schedule.resolved_max_steps)
        _bind_cache_materialization(writer, "train", train_cache)
    train_micro_steps = load_rank_micro_steps_from_cache(
        train_cache["cache_dir"],
        expected_fingerprint=str(train_cache["fingerprint"]),
        schedule=schedule,
        rank=int(accelerator.process_index),
        world_size=int(accelerator.num_processes),
    )
    train_micro_steps = _attach_image_processors_to_micro_steps(
        train_micro_steps,
        image_processor=_qwen_image_processor(components),
    )
    train_micro_steps = _apply_fa2_branch_proof_policy(train_micro_steps, config)
    loss_runner = LossRunner.from_config(config.losses)

    optimizer_group_plan = build_optimizer_group_plan(
        model,
        config.optimizer,
        adapter_receipt=adapter_result.receipt,
        special_token_receipt=special_token_result.receipt,
    )
    scheduler_plan = build_scheduler_plan(
        config.optimizer,
        total_training_steps=schedule.resolved_max_steps,
    )
    del scheduler_plan
    optimizer, scheduler = build_optimizer_and_scheduler(
        config.optimizer,
        optimizer_group_plan,
        total_training_steps=schedule.resolved_max_steps,
    )
    build_trainable_surface_receipt(
        model,
        adapter_receipt=adapter_result.receipt,
        special_token_receipt=special_token_result.receipt,
        optimizer_group_plan=optimizer_group_plan,
    )

    runtime = TrainRuntime(
        runtime_config=config.runtime,
        runtime_batch=schedule.runtime_batch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_mixed_precision=config.training.precision,
        max_grad_norm=config.training.max_grad_norm,
        accelerator=accelerator,
        rank_report_gatherer=_build_rank_report_gatherer(
            int(accelerator.num_processes)
        ),
    )

    eval_cache = _resolve_eval_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        accelerator=accelerator,
        rank=int(accelerator.process_index),
    )
    if eval_cache is None:
        eval_micro_steps = train_micro_steps
    else:
        eval_micro_steps = load_all_micro_steps_from_cache(
            eval_cache["cache_dir"],
            expected_fingerprint=str(eval_cache["fingerprint"]),
        )
        eval_micro_steps = _attach_image_processors_to_micro_steps(
            eval_micro_steps,
            image_processor=_qwen_image_processor(components),
        )
        if writer is not None:
            _bind_cache_materialization(writer, "eval", eval_cache)
    eval_micro_steps = _apply_fa2_branch_proof_policy(eval_micro_steps, config)
    checkpoint_writer = CheckpointWriter(run_directory.run_dir)
    eval_by_step: dict[int, dict[str, Any]] = {}
    committed_checkpoint_steps: set[int] = set()
    checkpoint_handler = _checkpoint_handler(
        checkpoint_writer,
        model=runtime.model,
        runtime=runtime,
        adapter_name=str(getattr(adapter_result.receipt, "adapter_name", "default")),
        special_token_result=special_token_result,
        schedule=schedule,
        base_model_path=components.base_model_path,
        base_config_sha256=components.base_config_sha256,
        tokenizer_sha256=components.tokenizer_sha256,
        writer=writer,
        eval_by_step=eval_by_step,
        committed_steps=committed_checkpoint_steps,
        lifecycle=lifecycle,
        save_final=config.checkpoint.save_final,
    )
    trainer = SupervisedTrainer(
        model=model,
        schedule=schedule,
        pack_stream=iter(train_micro_steps),
        loss_runner=loss_runner,
        runtime=runtime,
        on_completed_step=_train_logging_handler(writer, lifecycle, runtime),
        on_checkpoint=checkpoint_handler,
        on_eval=_eval_forward_handler(
                model=runtime.model,
                runtime=runtime,
                eval_micro_steps=eval_micro_steps,
                loss_runner=loss_runner,
                writer=writer,
                eval_source=config.data.eval.model_dump(mode="json")
                if config.data.eval is not None
                else None,
                eval_by_step=eval_by_step,
            ),
        on_final=_final_handler(
            checkpoint_handler=checkpoint_handler,
            committed_steps=committed_checkpoint_steps,
            save_final=config.checkpoint.save_final,
        ),
    )
    try:
        result = trainer.run()
        latest = result.latest_observation
        if writer is not None:
            writer.finalize(
                status="completed",
                updated_at=datetime.now(UTC).isoformat(),
                completed_steps=result.completed_steps,
                consumed_packs=result.consumed_micro_steps,
                checkpoint_event_count=result.scheduled_event_counts.get(
                    "checkpoint", 0
                ),
                optimizer_update_status=None
                if latest is None
                else latest.optimizer_update_status,
                finite_status=None if latest is None else latest.finite_status,
            )
        return {
            "run_dir": str(run_directory.run_dir),
            "run_id": run_id,
            "resolved_config_fingerprint": resolved_config.fingerprint,
            "completed_steps": result.completed_steps,
            "consumed_micro_steps": result.consumed_micro_steps,
            "scheduled_event_counts": dict(result.scheduled_event_counts),
        }
    finally:
        close_rank_report_gatherer = getattr(
            getattr(runtime, "rank_report_gatherer", None),
            "close",
            None,
        )
        if callable(close_rank_report_gatherer):
            close_rank_report_gatherer()


def build_base_micro_steps(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    materialization_workers: int | None = None,
) -> tuple[SupervisedMicroStep, ...]:
    return _build_micro_steps_for_dataset(
        config,
        components,
        vocab_groups,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        materialization_workers=materialization_workers,
    )


def _resolve_or_build_train_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    accelerator: Any,
    rank: int = 0,
) -> dict[str, Any]:
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
        accelerator=accelerator,
        rank=rank,
        build_micro_steps=lambda workers: build_base_micro_steps(
            config,
            components,
            vocab_groups,
            materialization_workers=workers,
        ),
    )


def _resolve_eval_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    accelerator: Any,
    rank: int = 0,
) -> dict[str, Any] | None:
    if config.data.eval is None:
        return None
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.eval,
        split="eval.forward",
        accelerator=accelerator,
        rank=rank,
        build_micro_steps=lambda workers: _build_micro_steps_for_dataset(
            config,
            components,
            vocab_groups,
            dataset=config.data.eval,
            split="eval.forward",
            materialization_workers=workers,
        ),
    )


def _resolve_or_build_pack_cache(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    repo_root: Path,
    dataset: Any,
    split: str,
    accelerator: Any,
    rank: int = 0,
    build_micro_steps: Callable[[int], Sequence[SupervisedMicroStep]],
    materialization_workers: int | None = None,
) -> dict[str, Any]:
    cache_root = Path(
        os.environ.get(
            "COORDEXP_SWIFT_PACK_CACHE_ROOT",
            str(repo_root / ".cache" / "coordexp_swift" / "packing"),
        )
    )
    world_size = 1 if accelerator is None else int(accelerator.num_processes)
    fingerprint = build_packing_cache_fingerprint(
        config,
        components,
        dataset=dataset,
        split=split,
    )
    determinants = build_packing_cache_determinants(
        config,
        components,
        dataset=dataset,
        split=split,
    )
    resolved_materialization_workers = _resolve_pack_cache_materialization_workers(
        materialization_workers
    )
    materialization = build_packing_cache_materialization(
        workers=resolved_materialization_workers,
        strategy=PACKING_CACHE_MATERIALIZATION_STRATEGY,
    )
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    cache_complete_before = False
    try:
        try:
            manifest = load_cache_manifest(
                cache_dir, expected_fingerprint=fingerprint
            )
            cache_complete_before = True
        except PackingCacheInvalidError:
            if world_size > 1:
                raise RuntimeContractError(
                    "distributed training requires prepared packing caches; run "
                    "`python -m src.prepare_train_cache --config <path>` before "
                    "`accelerate launch`",
                    code="training.pack_cache_not_prepared",
                    context={
                        "rank": rank,
                        "cache_dir": str(cache_dir),
                        "fingerprint": fingerprint,
                        "world_size": world_size,
                    },
                )
            micro_steps = tuple(build_micro_steps(resolved_materialization_workers))
            manifest = write_micro_step_cache(
                cache_dir,
                micro_steps,
                fingerprint=fingerprint,
                determinants=determinants,
                materialization=materialization,
                augmentation=_augmentation_receipt_from_micro_steps(micro_steps),
            )
    except RuntimeContractError:
        raise
    except BaseException as exc:
        raise RuntimeContractError(
            "failed to resolve packing cache",
            code="training.pack_cache_resolution_failed",
            context={
                "rank": rank,
                "cache_dir": str(cache_dir),
                "fingerprint": fingerprint,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1024],
            },
        ) from exc
    cache_manifest_path = manifest_path(cache_dir)
    return {
        "cache_dir": cache_dir,
        "format_version": str(manifest["version"]),
        "fingerprint": fingerprint,
        "micro_step_count": int(manifest["micro_step_count"]),
        "chunk_count": len(manifest["chunks"]),
        "chunk_size": int(manifest["chunk_size"]),
        "status": manifest["status"],
        "build_status": (
            "waited" if rank != 0 else ("hit" if cache_complete_before else "built")
        ),
        "manifest_path": cache_manifest_path,
        "manifest_sha256": _file_sha256(cache_manifest_path),
        "determinants_sha256": _sha256_json(manifest["determinants"]),
        "chunk_sha256s": [str(chunk["sha256"]) for chunk in manifest["chunks"]],
        "materialization": manifest.get("materialization"),
        "augmentation": manifest.get("augmentation"),
    }


def _pack_cache_preparation_receipt(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(cache["status"]),
        "build_status": str(cache["build_status"]),
        "cache_dir": str(cache["cache_dir"]),
        "format_version": str(cache["format_version"]),
        "fingerprint": str(cache["fingerprint"]),
        "manifest_path": str(cache["manifest_path"]),
        "manifest_sha256": str(cache["manifest_sha256"]),
        "micro_step_count": int(cache["micro_step_count"]),
    }


def _bind_cache_materialization(
    writer: RunWriter, split: str, cache: Mapping[str, Any]
) -> None:
    writer.bind_materialization(
        split,
        cache_format_version=str(cache["format_version"]),
        semantic_fingerprint=str(cache["fingerprint"]),
        determinant_digest=str(cache["determinants_sha256"]),
    )


def _attach_image_processors_to_micro_steps(
    micro_steps: Sequence[SupervisedMicroStep],
    *,
    image_processor: Any,
) -> tuple[SupervisedMicroStep, ...]:
    if image_processor is None:
        raise RuntimeContractError(
            "cached Qwen image encodings require a runtime image_processor",
            code="training.qwen_image_processor_missing",
            context={},
        )
    return tuple(
        replace(
            micro_step,
            encoded_examples=tuple(
                _attach_image_processor_to_encoded_example(
                    encoded_example,
                    image_processor=image_processor,
                )
                for encoded_example in micro_step.encoded_examples
            ),
        )
        for micro_step in micro_steps
    )


def _attach_image_processor_to_encoded_example(
    encoded_example: Any,
    *,
    image_processor: Any,
) -> Any:
    image_encoding = getattr(encoded_example, "image_encoding", None)
    if not isinstance(image_encoding, QwenImageEncoding):
        return encoded_example
    return replace(
        encoded_example,
        image_encoding=attach_qwen_image_processor(image_encoding, image_processor),
    )


def _apply_fa2_branch_proof_policy(
    micro_steps: Sequence[SupervisedMicroStep],
    config: Any,
) -> tuple[SupervisedMicroStep, ...]:
    policy = getattr(config.model, "fa2_branch_proof", "every_forward")
    configured: list[SupervisedMicroStep] = []
    for local_index, micro_step in enumerate(micro_steps):
        capture = policy == "every_forward" or (
            policy == "first_micro_step" and local_index == 0
        )
        configured.append(
            replace(
                micro_step,
                fa2_branch_evidence=None,
                capture_fa2_branch=capture,
                require_fa2_branch_proof=capture,
                fa2_branch_proof_policy=policy,
            )
        )
    return tuple(configured)


def _qwen_image_processor(components: Any) -> Any:
    processor = getattr(components, "processor", None)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeContractError(
            "Qwen components must expose processor.image_processor for lazy image packing",
            code="training.qwen_image_processor_missing",
            context={"processor_type": type(processor).__name__},
        )
    return image_processor


def _build_encoded_examples_for_dataset(
    config: Any,
    components: Any,
    raw_examples: Sequence[Any],
    *,
    materialization_workers: int | None = None,
) -> tuple[Any, ...]:
    workers = _resolve_pack_cache_materialization_workers(materialization_workers)
    if workers == 1:
        return tuple(
            _render_and_encode_example(
                raw_example,
                config=config,
                components=components,
            )
            for raw_example in raw_examples
        )
    return _encode_examples_with_fork_process_pool(
        config,
        components,
        raw_examples,
        workers=workers,
    )


def _encode_examples_with_fork_process_pool(
    config: Any,
    components: Any,
    raw_examples: Sequence[Any],
    *,
    workers: int,
) -> tuple[Any, ...]:
    mp_context = _fork_multiprocessing_context(workers)
    global _PACK_CACHE_WORKER_CONTEXT
    _PACK_CACHE_WORKER_CONTEXT = {
        "config": config,
        "components": components,
        "raw_examples": tuple(raw_examples),
    }
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(_encode_example_worker, index)
                for index in range(len(raw_examples))
            ]
            indexed_results = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
    finally:
        _PACK_CACHE_WORKER_CONTEXT = None
    return _restore_encoded_example_order(
        indexed_results,
        expected_count=len(raw_examples),
    )


def _encode_example_worker(index: int) -> tuple[int, Any]:
    context = _PACK_CACHE_WORKER_CONTEXT
    if context is None:
        raise RuntimeContractError(
            "packing cache worker context was not initialized",
            code="training.pack_cache_worker_context_missing",
            context={"index": index},
        )
    raw_examples = context["raw_examples"]
    raw_example = raw_examples[index]
    encoded = _render_and_encode_example(
        raw_example,
        config=context["config"],
        components=context["components"],
    )
    return index, encoded


def _render_and_encode_example(
    raw_example: Any,
    *,
    config: Any,
    components: Any,
) -> Any:
    rendered = render_example(
        raw_example,
        config.template,
        object_order_seed=_object_order_seed(config, raw_example.example_id),
    )
    return encode_rendered_example(
        raw_example,
        rendered,
        components=components,
        processor_config=config.model.processor,
        global_max_length=config.packing.global_max_length,
        materialize_image_pixels=False,
    )


def _restore_encoded_example_order(
    indexed_results: Sequence[tuple[int, Any]],
    *,
    expected_count: int,
) -> tuple[Any, ...]:
    ordered: list[Any | None] = [None for _ in range(expected_count)]
    seen: set[int] = set()
    for index, encoded_example in indexed_results:
        if index < 0 or index >= expected_count:
            raise RuntimeContractError(
                "packing cache worker returned an out-of-range example index",
                code="training.pack_cache_worker_index",
                context={"index": index, "expected_count": expected_count},
            )
        if index in seen:
            raise RuntimeContractError(
                "packing cache worker returned a duplicate example index",
                code="training.pack_cache_worker_index_duplicate",
                context={"index": index},
            )
        seen.add(index)
        ordered[index] = encoded_example
    if len(seen) != expected_count:
        missing = sorted(set(range(expected_count)) - seen)
        raise RuntimeContractError(
            "packing cache workers did not return every encoded example",
            code="training.pack_cache_worker_index_missing",
            context={"missing_indices": missing[:16], "missing_count": len(missing)},
        )
    return tuple(encoded_example for encoded_example in ordered)


def _resolve_pack_cache_materialization_workers(workers: int | None) -> int:
    resolved_workers = (
        DEFAULT_PACK_CACHE_MATERIALIZATION_WORKERS
        if workers is None
        else workers
    )
    if isinstance(resolved_workers, bool) or not isinstance(resolved_workers, int):
        raise RuntimeContractError(
            "packing cache materialization workers must be an integer",
            code="training.pack_cache_workers_invalid",
            context={"workers": resolved_workers},
        )
    if resolved_workers <= 0:
        raise RuntimeContractError(
            "packing cache materialization workers must be positive",
            code="training.pack_cache_workers_invalid",
            context={"workers": resolved_workers},
        )
    return resolved_workers


def _fork_multiprocessing_context(workers: int) -> Any:
    available_start_methods = tuple(multiprocessing.get_all_start_methods())
    if "fork" not in available_start_methods:
        raise RuntimeContractError(
            "parallel packing-cache materialization requires multiprocessing fork",
            code="training.pack_cache_workers_unavailable",
            context={
                "workers": workers,
                "available_start_methods": available_start_methods,
            },
        )
    try:
        return multiprocessing.get_context("fork")
    except ValueError as exc:
        raise RuntimeContractError(
            "parallel packing-cache materialization could not acquire fork context",
            code="training.pack_cache_workers_unavailable",
            context={
                "workers": workers,
                "available_start_methods": available_start_methods,
            },
            cause=exc,
        ) from exc


def _build_micro_steps_for_dataset(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    dataset: Any,
    split: str,
    materialization_workers: int | None = None,
) -> tuple[SupervisedMicroStep, ...]:
    if dataset is None:
        raise RuntimeContractError(
            "micro-step construction requires an explicit dataset split",
            code="training.dataset_split_missing",
            context={"split": split},
        )
    augmentation_result = _materialize_raw_examples_for_dataset(
        config,
        dataset,
        split=split,
    )
    raw_examples = augmentation_result.examples
    encoded_examples = _build_encoded_examples_for_dataset(
        config,
        components,
        raw_examples,
        materialization_workers=materialization_workers,
    )
    packs = plan_packed_sequences(
        encoded_examples,
        global_max_length=config.packing.global_max_length,
    )
    supervision = build_packed_supervision(packs, encoded_examples)
    token_atoms_by_pack = index_token_atoms_by_pack(supervision)
    micro_steps: list[SupervisedMicroStep] = []
    for pack in packs:
        pack_examples = _encoded_examples_for_pack(pack, encoded_examples)
        position_inputs = build_qwen_position_inputs(
            pack,
            pack_examples,
            image_token_id=_image_token_id(components),
        )
        token_sequence = build_token_sequence_from_packed_supervision(
            pack,
            token_atoms_by_pack.get(pack.pack_index, ()),
        )
        micro_steps.append(
            SupervisedMicroStep(
                pack=pack,
                encoded_examples=pack_examples,
                position_inputs=position_inputs,
                token_sequence=token_sequence,
                vocab_groups=vocab_groups,
                metadata={
                    "split": split,
                    "pack_id": pack.pack_index,
                    "example_ids": [segment.example_id for segment in pack.segments],
                    "augmentation_receipt": augmentation_result.receipt,
                },
                expected_vocab_size=components.token_identity.tokenizer_vocab_size,
                fa2_model_dtype=config.training.precision,
                capture_fa2_branch=config.model.fa2_branch_proof == "every_forward",
                require_fa2_branch_proof=config.model.fa2_branch_proof
                == "every_forward",
            )
        )
    if not micro_steps:
        raise RuntimeContractError(
            "micro-step construction produced no packs",
            code="training.empty_micro_step_plan",
            context={"split": split},
        )
    return tuple(micro_steps)


def _materialize_raw_examples_for_dataset(
    config: Any,
    dataset: Any,
    *,
    split: str,
) -> AugmentationMaterializationResult:
    raw_examples = load_raw_examples(dataset)
    processor = build_augmentation_processor(config, split=split)
    return processor.materialize(
        raw_examples,
        split=split,
        object_ordering=config.template.object_ordering,
    )


def enable_training_memory_savers(model: Any) -> dict[str, Any]:
    train_mode_enabled = _enable_train_mode(model)
    use_cache_disabled = _disable_use_cache(model)
    gradient_checkpointing_kwargs = {"use_reentrant": False}
    gradient_checkpointing_enabled, applied_gradient_checkpointing_kwargs = _enable_gradient_checkpointing(
        model,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )
    input_require_grads_enabled = _call_first_available(
        model,
        "enable_input_require_grads",
    )
    return {
        "train_mode_enabled": train_mode_enabled,
        "model_training": _model_training_state(model),
        "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
        "gradient_checkpointing_kwargs": applied_gradient_checkpointing_kwargs,
        "input_require_grads_enabled": input_require_grads_enabled,
        "use_cache_disabled": use_cache_disabled,
    }


def _checkpoint_handler(
    checkpoint_writer: CheckpointWriter,
    *,
    model: Any,
    runtime: Any | None = None,
    adapter_name: str,
    special_token_result: Any,
    schedule: ResolvedStepSchedule,
    base_model_path: Path,
    base_config_sha256: str,
    tokenizer_sha256: str,
    writer: RunWriter | None,
    eval_by_step: Mapping[int, Mapping[str, Any]],
    committed_steps: set[int],
    lifecycle: dict[str, Any],
    save_final: bool = True,
) -> Any:
    def handle(scheduled_event: Any, observation: CompletedStepObservation) -> None:
        accelerator = getattr(runtime, "accelerator", runtime)
        step = int(scheduled_event.planned_step_id)
        eval_observation = eval_by_step.get(step)
        checkpoint_writer.write_checkpoint(
            step=step,
            accelerator=accelerator,
            model=model,
            adapter_name=adapter_name,
            special_token_result=special_token_result,
            base_model_path=base_model_path,
            base_config_sha256=base_config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            run_writer=writer,
            is_final=save_final and step == schedule.resolved_max_steps,
            best_candidate=None if eval_observation is None else {
                "completed": True,
                "selector": BEST_EVAL_SELECTOR_NAME,
                "value": eval_observation.get(BEST_EVAL_SELECTOR_NAME),
                "optimizer_update_status": observation.optimizer_update_status,
                "finite_status": observation.finite_status,
            },
        )
        committed_steps.add(step)
        lifecycle["checkpoint_event_count"] = int(
            lifecycle.get("checkpoint_event_count", 0)
        ) + 1

    return handle


def _template_identity(config: Any) -> dict[str, Any]:
    template = config.template
    if hasattr(template, "model_dump"):
        return dict(template.model_dump(mode="json"))
    return dict(template)


def _unwrap_checkpoint_model(model: Any, *, runtime: Any | None) -> Any:
    if runtime is None:
        return model
    accelerator = getattr(runtime, "accelerator", None)
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        return unwrap_model(model)
    return model


def _eval_forward_handler(
    *,
    model: Any,
    runtime: Any,
    eval_micro_steps: tuple[SupervisedMicroStep, ...],
    loss_runner: LossRunner,
    writer: RunWriter | None,
    eval_source: dict[str, Any] | None,
    eval_by_step: dict[int, dict[str, Any]],
) -> Any:
    def handle(scheduled_event: Any, observation: CompletedStepObservation) -> None:
        result = ForwardEvalRunner(
            model=model,
            micro_step_stream=iter(eval_micro_steps),
            loss_runner=loss_runner,
            eval_source=eval_source,
            runtime=runtime,
        ).run(
            planned_step_id=scheduled_event.planned_step_id,
            trigger_reasons=scheduled_event.trigger_reasons,
        )
        row = result.to_logging_row()
        row["optimizer_update_status"] = observation.optimizer_update_status
        row["finite_status"] = observation.finite_status
        eval_by_step[int(scheduled_event.planned_step_id)] = row
        _append_logging_row_shared(writer=writer, row=row, runtime=runtime)

    return handle


def _final_handler(
    *, checkpoint_handler: Any, committed_steps: set[int], save_final: bool = True
) -> Any:
    def handle(scheduled_event: Any, observation: CompletedStepObservation) -> None:
        if save_final and int(scheduled_event.planned_step_id) not in committed_steps:
            checkpoint_handler(scheduled_event, observation)

    return handle


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _augmentation_receipt_from_micro_steps(
    micro_steps: Sequence[SupervisedMicroStep],
) -> dict[str, Any] | None:
    for micro_step in micro_steps:
        metadata = getattr(micro_step, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        receipt = metadata.get("augmentation_receipt")
        if isinstance(receipt, Mapping):
            return dict(receipt)
    return None


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _resolve_shared_run_directory(
    config: Any,
    *,
    cwd: Path,
    accelerator: Any,
) -> RunDirectory:
    """Resolve one rank-zero run path and share its compact descriptor."""

    descriptor: dict[str, Any] | None = None
    if bool(accelerator.is_main_process):
        try:
            selected = resolve_run_directory(config, cwd=cwd)
            descriptor = {
                "ok": True,
                "run_name": selected.run_name,
                "artifact_root": str(selected.artifact_root),
                "run_dir": str(selected.run_dir),
                "collision_policy": selected.collision_policy,
            }
        except Exception as exc:
            descriptor = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    objects: list[Any] = [descriptor]
    if int(accelerator.num_processes) > 1:
        if broadcast_object_list is None:
            raise RuntimeContractError(
                "distributed run descriptor broadcast requires accelerate",
                code="runtime.run_descriptor_broadcast_unavailable",
            )
        broadcast_object_list(objects, from_process=0)
    shared = objects[0]
    if not isinstance(shared, Mapping):
        raise RuntimeContractError(
            "run descriptor broadcast returned an invalid payload",
            code="runtime.run_descriptor_invalid",
        )
    if not bool(shared.get("ok")):
        raise RuntimeContractError(
            "rank zero failed to resolve the shared run directory",
            code="runtime.run_directory_resolution_failed",
            context={
                "error_type": str(shared.get("error_type", "unknown")),
                "error": str(shared.get("error", "unknown error")),
            },
        )
    return RunDirectory(
        run_name=str(shared["run_name"]),
        artifact_root=Path(str(shared["artifact_root"])),
        run_dir=Path(str(shared["run_dir"])),
        collision_policy=str(shared["collision_policy"]),
    )


def _loss_plan_artifact(config: Any, vocab_groups_artifact: dict[str, Any]) -> dict[str, Any]:
    term_order = ["base_ce", "token_type_gate"]
    if config.losses.protected.coord_gaussian_rps.weight > 0.0:
        term_order.append("coord_gaussian_rps")
    return {
        "normalizer": config.losses.normalizer,
        "objective_dtype": "float32_selected_logits",
        "protected": config.losses.protected.model_dump(mode="json"),
        "term_order": term_order,
        "vocabulary_groups": vocab_groups_artifact,
        "finite_policy": {
            "pre_backward_scalar_gate": "all_rank_consensus",
            "post_backward_gradient_gate": "all_rank_consensus",
        },
        "metric_definitions": {
            "top_level": ["acc_top1", "acc_top5"],
            "weighted_losses": [
                *(f"loss/{term_name}" for term_name in term_order),
                "loss/total",
            ],
            "counts": [
                "count/supervised_atoms",
                "count/eligible_segments",
                "count/skipped_segments",
                "count/packs",
                "count/examples",
            ],
        },
    }


def _encoded_examples_for_pack(
    pack: PackedSequence,
    encoded_examples: Sequence[Any],
) -> tuple[Any, ...]:
    examples_by_id = {
        str(getattr(example, "example_id")): example
        for example in encoded_examples
    }
    return tuple(examples_by_id[segment.example_id] for segment in pack.segments)


def _object_order_seed(config: Any, example_id: str) -> int | None:
    del example_id
    if config.template.object_ordering == "random":
        return int(config.runtime.seed)
    return None


def _image_token_id(components: Any) -> int | None:
    tokenizer = getattr(components, "tokenizer", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    token_id = convert("<|image_pad|>")
    return None if token_id is None else int(token_id)


def _disable_use_cache(model: Any) -> list[str]:
    disabled: list[str] = []
    seen_config_ids: set[int] = set()
    for owner_name, owner in _model_and_base_model_owners(model):
        for config_path, config in _config_owners(owner):
            config_id = id(config)
            if config_id in seen_config_ids:
                continue
            seen_config_ids.add(config_id)
            if not hasattr(config, "use_cache"):
                continue
            if getattr(config, "use_cache") is not False:
                setattr(config, "use_cache", False)
            disabled.append(f"{owner_name}.{config_path}")
    return disabled


def _build_accelerator(training_precision: str) -> Any:
    if Accelerator is None:
        raise RuntimeContractError(
            "training requires the accelerate package",
            code="runtime.accelerate_unavailable",
        )
    kwargs: dict[str, Any] = {
        # CoordExp-Swift owns planned-step loss normalization and optimizer
        # cadence. Accelerate's accumulation counter would additionally divide
        # loss inside accelerator.backward(), so keep it neutral and use
        # TrainRuntime.no_sync for intermediate microsteps.
        "gradient_accumulation_steps": 1,
        "mixed_precision": str(training_precision),
    }
    return Accelerator(**kwargs)


def _rank_report_value(report: Any, name: str, default: Any) -> Any:
    if isinstance(report, Mapping):
        return report.get(name, default)
    return getattr(report, name, default)


def _rank_report_kind(report: Any) -> str:
    if isinstance(report, Mapping):
        return f"mapping:{report.get('kind', 'unspecified')}"
    report_type = type(report)
    return f"{report_type.__module__}:{report_type.__qualname__}"


def _rank_report_digest(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def _rank_report_header(
    report: Any,
    *,
    sequence: int,
    payload: bytes,
    serialization_status: int,
) -> bytes:
    return _RANK_REPORT_HEADER.pack(
        _RANK_REPORT_MAGIC,
        int(sequence),
        int(_rank_report_value(report, "planned_step_id", 0)),
        int(_rank_report_value(report, "rank", 0)),
        int(_rank_report_value(report, "world_size", 1)),
        int(serialization_status),
        len(payload),
        _rank_report_digest(_rank_report_kind(report)),
        _rank_report_digest(str(_rank_report_value(report, "split", ""))),
        zlib.crc32(payload),
    )


def _unpack_rank_report_header(header: bytes) -> dict[str, int | bytes]:
    (
        magic,
        sequence,
        planned_step_id,
        rank,
        world_size,
        serialization_status,
        payload_size,
        kind_digest,
        split_digest,
        payload_crc32,
    ) = _RANK_REPORT_HEADER.unpack(header)
    return {
        "magic": magic,
        "sequence": sequence,
        "planned_step_id": planned_step_id,
        "rank": rank,
        "world_size": world_size,
        "serialization_status": serialization_status,
        "payload_size": payload_size,
        "kind_digest": kind_digest,
        "split_digest": split_digest,
        "payload_crc32": payload_crc32,
    }


def _all_gather_cpu_bytes(
    distributed: Any,
    payload: bytes,
    *,
    width: int,
    world_size: int,
    group: Any | None,
) -> tuple[bytes, ...]:
    local = torch.zeros(width, dtype=torch.uint8, device="cpu")
    if payload:
        if len(payload) > width:
            raise ValueError("payload exceeds fixed collective width")
        local[: len(payload)] = torch.tensor(tuple(payload), dtype=torch.uint8)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    distributed.all_gather(gathered, local, group=group)
    return tuple(bytes(item.tolist()) for item in gathered)


def _validate_rank_report_headers(
    headers: Sequence[bytes],
    *,
    sequence: int,
    world_size: int,
) -> tuple[dict[str, int | bytes], ...]:
    unpacked = tuple(_unpack_rank_report_header(header) for header in headers)
    if any(header["magic"] != _RANK_REPORT_MAGIC for header in unpacked):
        raise RuntimeContractError(
            "rank report control headers have invalid framing",
            code="runtime.report_gather_framing",
            context={"sequence": sequence},
        )
    observed_sequences = sorted({int(header["sequence"]) for header in unpacked})
    if observed_sequences != [sequence]:
        raise RuntimeContractError(
            "rank report collectives entered in different sequence order",
            code="runtime.report_gather_sequence",
            context={
                "expected_sequence": sequence,
                "observed_sequences": observed_sequences,
            },
        )
    observed_ranks = sorted(int(header["rank"]) for header in unpacked)
    observed_world_sizes = sorted({int(header["world_size"]) for header in unpacked})
    if observed_ranks != list(range(world_size)) or observed_world_sizes != [
        world_size
    ]:
        raise RuntimeContractError(
            "rank report control headers disagree on distributed identity",
            code="runtime.report_gather_ranks",
            context={
                "expected_world_size": world_size,
                "observed_ranks": observed_ranks,
                "observed_world_sizes": observed_world_sizes,
            },
        )
    identity_fields = ("planned_step_id", "kind_digest", "split_digest")
    disagreements = {
        field: sorted({int(header[field]) for header in unpacked})
        for field in identity_fields
        if len({int(header[field]) for header in unpacked}) != 1
    }
    if disagreements:
        raise RuntimeContractError(
            "rank report control headers disagree on report identity",
            code="runtime.report_gather_identity",
            context={"sequence": sequence, "disagreements": disagreements},
        )
    failed_serialization_ranks = [
        int(header["rank"])
        for header in unpacked
        if int(header["serialization_status"]) != 0
    ]
    if failed_serialization_ranks:
        raise RuntimeContractError(
            "one or more ranks could not serialize a rank report",
            code="runtime.report_serialize_failed",
            context={
                "sequence": sequence,
                "failed_ranks": failed_serialization_ranks,
            },
        )
    oversized_ranks = [
        int(header["rank"])
        for header in unpacked
        if int(header["payload_size"]) > _RANK_REPORT_MAX_PAYLOAD_BYTES
    ]
    if oversized_ranks:
        raise RuntimeContractError(
            "rank report payload exceeds the bounded control-plane limit",
            code="runtime.report_gather_size",
            context={
                "sequence": sequence,
                "max_payload_bytes": _RANK_REPORT_MAX_PAYLOAD_BYTES,
                "oversized_ranks": oversized_ranks,
            },
        )
    return unpacked


def _build_rank_report_gatherer(world_size: int) -> Any | None:
    if world_size <= 1:
        return None

    distributed = torch.distributed
    control_group: Any | None = None
    control_group_ready = False
    sequence = 0

    def ensure_control_group() -> Any | None:
        nonlocal control_group, control_group_ready
        if control_group_ready:
            return control_group
        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeContractError(
                "multi-rank finite gates require initialized torch.distributed",
                code="runtime.distributed_gather_uninitialized",
                context={"world_size": world_size},
            )
        observed_world_size = int(distributed.get_world_size())
        if observed_world_size != world_size:
            raise RuntimeContractError(
                "rank report gatherer world size disagrees with torch.distributed",
                code="runtime.report_gather_ranks",
                context={
                    "expected_world_size": world_size,
                    "observed_world_size": observed_world_size,
                },
            )
        backend = str(distributed.get_backend()).lower()
        if "gloo" not in backend:
            is_gloo_available = getattr(distributed, "is_gloo_available", None)
            if callable(is_gloo_available) and not bool(is_gloo_available()):
                raise RuntimeContractError(
                    "bounded rank report gathering requires the gloo backend",
                    code="runtime.report_gather_backend",
                    context={"default_backend": backend},
                )
            control_group = distributed.new_group(
                ranks=list(range(world_size)),
                backend="gloo",
                timeout=timedelta(seconds=_RANK_REPORT_CONTROL_TIMEOUT_SECONDS),
            )
        control_group_ready = True
        return control_group

    def gather(local_report: Any) -> tuple[Any, ...]:
        nonlocal sequence
        group = ensure_control_group()
        sequence += 1
        serialization_status = 0
        try:
            payload = pickle.dumps(local_report, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException:
            payload = b""
            serialization_status = 1
        local_header = _rank_report_header(
            local_report,
            sequence=sequence,
            payload=payload,
            serialization_status=serialization_status,
        )
        local_frame = local_header + payload[:_RANK_REPORT_MAX_PAYLOAD_BYTES]
        gathered_frames = _all_gather_cpu_bytes(
            distributed,
            local_frame,
            width=_RANK_REPORT_FRAME_BYTES,
            world_size=world_size,
            group=group,
        )
        headers = _validate_rank_report_headers(
            tuple(frame[: _RANK_REPORT_HEADER.size] for frame in gathered_frames),
            sequence=sequence,
            world_size=world_size,
        )
        reports: list[Any] = []
        for header, gathered_frame in zip(headers, gathered_frames, strict=True):
            payload_size = int(header["payload_size"])
            framed_payload = gathered_frame[
                _RANK_REPORT_HEADER.size : _RANK_REPORT_HEADER.size + payload_size
            ]
            if zlib.crc32(framed_payload) != int(header["payload_crc32"]):
                raise RuntimeContractError(
                    "rank report payload checksum does not match its control header",
                    code="runtime.report_gather_framing",
                    context={
                        "sequence": sequence,
                        "rank": int(header["rank"]),
                    },
                )
            try:
                reports.append(pickle.loads(framed_payload))
            except BaseException as exc:
                raise RuntimeContractError(
                    "rank report payload could not be decoded",
                    code="runtime.report_gather_framing",
                    context={
                        "sequence": sequence,
                        "rank": int(header["rank"]),
                    },
                ) from exc
        return tuple(reports)

    def close() -> None:
        nonlocal control_group, control_group_ready
        if (
            control_group_ready
            and control_group is not None
            and distributed.is_available()
            and distributed.is_initialized()
        ):
            try:
                distributed.destroy_process_group(control_group)
            except BaseException:
                pass
        control_group = None
        control_group_ready = False

    gather.close = close  # type: ignore[attr-defined]
    return gather


def _config_owners(owner: Any) -> tuple[tuple[str, Any], ...]:
    config = getattr(owner, "config", None)
    if config is None:
        return ()
    owners: list[tuple[str, Any]] = [("config", config)]
    for nested_name in ("text_config", "language_config"):
        nested = getattr(config, nested_name, None)
        if nested is not None:
            owners.append((f"config.{nested_name}", nested))
    return tuple(owners)


def _enable_train_mode(model: Any) -> bool:
    train = getattr(model, "train", None)
    if not callable(train):
        return False
    train()
    return True


def _model_training_state(model: Any) -> bool | None:
    training = getattr(model, "training", None)
    if training is None:
        return None
    return bool(training)


def _enable_gradient_checkpointing(
    model: Any,
    *,
    gradient_checkpointing_kwargs: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    for _owner_name, owner in _model_and_base_model_owners(model):
        method = getattr(owner, "gradient_checkpointing_enable", None)
        if not callable(method):
            continue
        try:
            method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            return True, dict(gradient_checkpointing_kwargs)
        except TypeError:
            method()
            return True, None
    return False, None


def _call_first_available(model: Any, method_name: str) -> bool:
    for _owner_name, owner in _model_and_base_model_owners(model):
        method = getattr(owner, method_name, None)
        if callable(method):
            method()
            return True
    return False


def _model_and_base_model_owners(model: Any) -> tuple[tuple[str, Any], ...]:
    owners: list[tuple[str, Any]] = [("model", model)]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        try:
            base_model = get_base_model()
        except Exception:
            base_model = None
        if base_model is not None and base_model is not model:
            owners.append(("base_model", base_model))
    return tuple(owners)


def _run_id(run_name: str, fingerprint: str) -> str:
    return f"{run_name}-{fingerprint[:12]}"
