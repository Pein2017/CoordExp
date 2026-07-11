"""Training pipeline assembly for the V1 supervised smoke."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

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
from src.artifacts import MetricStreamEvent, RunArtifactManager
from src.artifacts.checkpoints import CheckpointWriter
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
    build_packing_cache_materialization,
    build_packing_cache_determinants,
    build_packing_cache_fingerprint,
    cache_dir_for_fingerprint,
    cache_is_complete,
    load_all_micro_steps_from_cache,
    load_cache_manifest,
    load_rank_micro_steps_from_cache,
    manifest_path,
    write_micro_step_cache,
)
from src.training.supervised_trainer import (
    ScheduledTrainerEvent,
    SupervisedMicroStep,
    SupervisedTrainer,
    SupervisedTrainerEvent,
)


TRAIN_SPLIT = "train"
_PACK_CACHE_WORKER_CONTEXT: dict[str, Any] | None = None
BEST_EVAL_SELECTOR_SPLIT = "eval.forward"
BEST_EVAL_SELECTOR_NAME = "acc_top1"
PROGRESS_EVENT_TYPES = frozenset(
    {
        "planned_step.started",
        "planned_step.prepared",
        "micro_step.forward",
        "micro_step.loss",
        "micro_step.pre_backward_gate",
        "micro_step.backward",
        "planned_step.loss",
        "planned_step.pre_backward_gate",
        "planned_step.post_backward_gate",
        "planned_step.completed",
    }
)


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


@dataclass
class TrainingArtifactBridge:
    manager: RunArtifactManager
    rank: int
    world_size: int

    def __call__(self, event: SupervisedTrainerEvent) -> None:
        if event.event_type in PROGRESS_EVENT_TYPES:
            self._append_progress_event(event)
        if event.event_type != "planned_step.completed":
            return
        payload = dict(event.payload)
        qwen_receipts = [
            dict(receipt)
            for receipt in payload.get("qwen_forward_receipts", ())
        ]
        if qwen_receipts:
            self.manager.write_receipt(
                f"forward_step_{event.planned_step_id}",
                {
                    "planned_step_id": event.planned_step_id,
                    "qwen_forward_receipts": qwen_receipts,
                },
                category="qwen",
            )
        loss_artifact = payload.get("loss_bundle")
        if loss_artifact is None:
            loss_artifact = payload.get("loss_bundle_artifact")
        if not isinstance(loss_artifact, dict):
            return
        metrics = loss_artifact.get("metrics")
        if not isinstance(metrics, dict):
            return
        optimizer_update_status = str(
            payload.get("optimizer_update_status", "unavailable")
        )
        finite_status = str(payload.get("finite_status", loss_artifact.get("finite_status", "unavailable")))
        scheduler_artifact = payload.get("scheduler")
        for name, value in _scheduler_lr_metrics(scheduler_artifact).items():
            self.manager.append_metric_event(
                MetricStreamEvent(
                    event_type="metric",
                    planned_step_id=event.planned_step_id,
                    split=TRAIN_SPLIT,
                    name=name,
                    value=value,
                    trigger_reasons=("planned_step.completed",),
                    optimizer_update_status=optimizer_update_status,
                    finite_status=finite_status,
                    warning_status="none",
                    rank=self.rank,
                    world_size=self.world_size,
                )
            )
        for name in sorted(metrics):
            value = metrics[name]
            self.manager.append_metric_event(
                MetricStreamEvent(
                    event_type="metric",
                    planned_step_id=event.planned_step_id,
                    split=TRAIN_SPLIT,
                    name=str(name),
                    value=None if value is None else float(value),
                    trigger_reasons=("planned_step.completed",),
                    optimizer_update_status=optimizer_update_status,
                    finite_status=finite_status,
                    warning_status="none",
                    rank=self.rank,
                    world_size=self.world_size,
                )
            )

    def _append_progress_event(self, event: SupervisedTrainerEvent) -> None:
        payload = dict(event.payload)
        record: dict[str, Any] = {
            "event_type": event.event_type,
            "planned_step_id": event.planned_step_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "monotonic_ns": time.monotonic_ns(),
        }
        if "local_micro_step_index" in payload:
            record["local_micro_step_index"] = int(payload["local_micro_step_index"])
        sync_gradients = payload.get("sync_gradients")
        if sync_gradients is not None:
            record["sync_gradients"] = bool(sync_gradients)
        optimizer_update_status = payload.get("optimizer_update_status")
        if optimizer_update_status is not None:
            record["optimizer_update_status"] = str(optimizer_update_status)
        finite_status = payload.get("finite_status")
        if finite_status is not None:
            record["finite_status"] = str(finite_status)
        stage = payload.get("stage")
        if stage is not None:
            record["stage"] = str(stage)
        timings = _timings_artifact(payload.get("timings_ns"))
        receipt = payload.get("receipt")
        if isinstance(receipt, Mapping):
            for key in ("pack_index", "pack_length", "segment_count"):
                if key in receipt:
                    record[key] = receipt[key]
            for key in (
                "pixel_values_shape",
                "output_logits_shape",
                "placeholder_token_count",
                "expected_visual_token_count",
            ):
                if key in receipt:
                    record[key] = receipt[key]
            receipt_timings = _timings_artifact(receipt.get("timings_ns"))
            if receipt_timings:
                timings.update(receipt_timings)
            fa2_varlen = receipt.get("fa2_varlen")
            if isinstance(fa2_varlen, Mapping):
                for key in ("max_length_q", "max_length_k", "segment_boundaries"):
                    if key in fa2_varlen:
                        record[f"fa2_{key}"] = fa2_varlen[key]
        if timings:
            record["timings_ns"] = timings
        output_path = (
            self.manager.run_dir
            / "diagnostics"
            / f"progress.rank-{self.rank}.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    allow_nan=False,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )


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


def _timings_artifact(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    timings: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, bool):
            timings[str(key)] = int(raw)
            continue
        if isinstance(raw, int):
            timings[str(key)] = raw
            continue
        if isinstance(raw, float) and raw.is_integer():
            timings[str(key)] = int(raw)
    return timings


@dataclass
class BestEvalMetricStore:
    _events_by_step: dict[int, MetricStreamEvent]

    def __init__(self) -> None:
        self._events_by_step = {}

    def record(self, event: MetricStreamEvent) -> None:
        if event.split != BEST_EVAL_SELECTOR_SPLIT or event.name != BEST_EVAL_SELECTOR_NAME:
            return
        self._events_by_step[event.planned_step_id] = event

    def best_for_step(self, planned_step_id: int) -> MetricStreamEvent | None:
        return self._events_by_step.get(int(planned_step_id))


@dataclass(frozen=True)
class _NonWritingArtifactBridge:
    """Temporary Wave-1 no-write bridge; Wave 2 deletes this legacy surface."""

    run_dir: Path
    run_id: str

    def append_metric_event(self, event: Any) -> None:
        del event

    def finalize(self, *, status: str, completed_at: str) -> None:
        del status, completed_at

    def write_receipt(self, name: str, payload: Any, *, category: str) -> None:
        del name, payload, category

    def write_report(self, name: str, payload: Any) -> None:
        del name, payload

    def write_resolved_config(self, resolved_config: Any) -> None:
        del resolved_config

    def write_schedule(self, schedule: Any) -> None:
        del schedule

    def write_eval_forward_summary(self, **payload: Any) -> Path:
        del payload
        return self.run_dir / "eval" / "non-main.json"


def _initialize_artifact_owner(
    *,
    accelerator: Any,
    run_directory: RunDirectory,
    run_id: str,
    created_at: str,
    repo_root: Path,
    resolved_config: Any,
) -> Any:
    manager: Any | None = None
    status: dict[str, Any] | None = None
    if bool(accelerator.is_main_process):
        try:
            manager = RunArtifactManager.initialize(
                run_directory=run_directory,
                run_id=run_id,
                created_at=created_at,
                runtime_identity={
                    "entrypoint": "python -m src.train",
                    "cwd": str(repo_root),
                },
                backend_status={"active": ["accelerate"]},
            )
            manager.write_resolved_config(resolved_config)
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
        return manager
    return _NonWritingArtifactBridge(run_directory.run_dir, run_id)


def _training_artifact_event_sink(manager: Any, runtime: Any) -> Any | None:
    if not bool(runtime.is_main_process):
        return None
    return TrainingArtifactBridge(
        manager,
        rank=runtime.rank,
        world_size=runtime.world_size,
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
    manager = _initialize_artifact_owner(
        accelerator=accelerator,
        run_directory=run_directory,
        run_id=run_id,
        created_at=created_at,
        repo_root=repo_root,
        resolved_config=resolved_config,
    )
    seed_receipt = seed_training_runtime(
        config.runtime.seed,
        deterministic=False,
        phase="pipeline_assembly",
    )
    manager.write_receipt(
        "seed_control",
        _artifact_dict(seed_receipt),
        category="runtime",
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
    manager.write_receipt(
        "setup",
        {
            "components": _artifact_dict(components),
            "runtime_controls": _artifact_dict(runtime_controls),
        },
        category="qwen",
    )

    adapter_evidence = load_default_adapter_source_gate_evidence(repo_root)
    adapter_plan = build_adapter_setup_plan(
        config.adapter,
        adapter_evidence,
        base_model_path=components.base_model_path,
    )
    manager.write_receipt(
        "adapter_setup_plan",
        adapter_plan.to_artifact_dict(),
        category="optimizer",
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
        special_token_load_receipt = load_special_token_embedding_deltas(
            special_token_result,
            adapter_plan.repaired_embedding_payload_path,
            expected_base_model_path=components.base_model_path,
            expected_base_config_sha256=components.base_config_sha256,
            expected_tokenizer_sha256=components.tokenizer_sha256,
        )
        manager.write_receipt(
            "special_token_embedding_seed",
            _artifact_dict(special_token_load_receipt),
            category="qwen",
        )
    manager.write_receipt(
        "memory_savers",
        enable_training_memory_savers(model),
        category="runtime",
    )

    vocab_groups = build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )
    vocab_groups_artifact = _artifact_dict(vocab_groups)
    manager.write_report("token_type_vocab", vocab_groups_artifact)
    train_cache = _resolve_or_build_train_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        rank=int(accelerator.process_index),
    )
    schedule = resolve_planned_step_schedule(
        config,
        packs_per_epoch=train_cache["micro_step_count"],
        world_size=int(accelerator.num_processes),
        source_config_path=str(resolved_config.entry_config_path),
    )
    manager.write_schedule(schedule)
    train_micro_steps = load_rank_micro_steps_from_cache(
        train_cache["cache_dir"],
        schedule=schedule,
        rank=int(accelerator.process_index),
        world_size=int(accelerator.num_processes),
    )
    train_micro_steps = _attach_image_processors_to_micro_steps(
        train_micro_steps,
        image_processor=_qwen_image_processor(components),
    )
    train_micro_steps = _apply_fa2_branch_proof_policy(train_micro_steps, config)
    manager.write_receipt(
        "pack_plan",
        _pack_plan_artifact(
            train_micro_steps,
            schedule=schedule,
            cache=train_cache,
        ),
        category="packing",
    )
    loss_runner = LossRunner.from_config(config.losses)
    manager.write_receipt(
        "loss_plan",
        _loss_plan_artifact(config, vocab_groups_artifact),
        category="losses",
    )

    optimizer_group_plan = build_optimizer_group_plan(
        model,
        config.optimizer,
        adapter_receipt=adapter_result.receipt,
        special_token_receipt=special_token_result.receipt,
    )
    manager.write_receipt(
        "optimizer_groups",
        optimizer_group_plan.to_artifact_dict(),
        category="optimizer",
    )
    scheduler_plan = build_scheduler_plan(
        config.optimizer,
        total_training_steps=schedule.resolved_max_steps,
    )
    manager.write_receipt(
        "scheduler_plan",
        scheduler_plan.to_artifact_dict(),
        category="optimizer",
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        config.optimizer,
        optimizer_group_plan,
        total_training_steps=schedule.resolved_max_steps,
    )
    trainable_surface = build_trainable_surface_receipt(
        model,
        adapter_receipt=adapter_result.receipt,
        special_token_receipt=special_token_result.receipt,
        optimizer_group_plan=optimizer_group_plan,
    )
    manager.write_receipt(
        "trainable_surface",
        trainable_surface.to_artifact_dict(),
        category="optimizer",
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
    manager.write_receipt(
        "runtime_setup",
        _artifact_dict(runtime.setup_receipt),
        category="runtime",
    )

    eval_cache = _resolve_eval_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        rank=int(accelerator.process_index),
    )
    if eval_cache is None:
        eval_micro_steps = train_micro_steps
    else:
        eval_micro_steps = load_all_micro_steps_from_cache(eval_cache["cache_dir"])
        eval_micro_steps = _attach_image_processors_to_micro_steps(
            eval_micro_steps,
            image_processor=_qwen_image_processor(components),
        )
        manager.write_receipt(
            "eval_pack_plan",
            _pack_plan_artifact(
                eval_micro_steps,
                schedule=schedule,
                cache=eval_cache,
            ),
            category="packing",
        )
    eval_micro_steps = _apply_fa2_branch_proof_policy(eval_micro_steps, config)
    checkpoint_writer = CheckpointWriter(manager)
    best_eval_metrics = BestEvalMetricStore()
    trainer = SupervisedTrainer(
        model=model,
        schedule=schedule,
        pack_stream=iter(train_micro_steps),
        loss_runner=loss_runner,
        runtime=runtime,
        event_sink=_training_artifact_event_sink(manager, runtime),
        scheduled_event_handlers={
            "checkpoint": _checkpoint_handler(
                checkpoint_writer,
                model=runtime.model,
                runtime=runtime,
                adapter_receipt=adapter_result.receipt,
                special_token_result=special_token_result,
                trainable_surface=trainable_surface,
                processor_identity=_artifact_dict(components.processor_identity),
                template_identity=_template_identity(config),
                resolved_config_fingerprint=resolved_config.fingerprint,
                schedule=schedule,
                base_model_path=components.base_model_path,
                base_config_sha256=components.base_config_sha256,
                tokenizer_sha256=components.tokenizer_sha256,
                best_eval_metrics=best_eval_metrics,
            ),
            "eval.forward": _eval_forward_handler(
                model=runtime.model,
                runtime=runtime,
                eval_micro_steps=eval_micro_steps,
                loss_runner=loss_runner,
                manager=manager,
                eval_source=config.data.eval.model_dump(mode="json")
                if config.data.eval is not None
                else None,
                best_eval_metrics=best_eval_metrics,
            ),
            "final": _final_handler(),
        },
    )
    result = trainer.run()
    manager.write_receipt(
        "training_result",
        result.to_artifact_dict(),
        category="runtime",
    )
    manager.finalize(status="completed", completed_at=datetime.now(UTC).isoformat())
    return {
        "run_dir": str(manager.run_dir),
        "run_id": manager.run_id,
        "resolved_config_fingerprint": resolved_config.fingerprint,
        "completed_steps": result.completed_steps,
        "consumed_micro_steps": result.consumed_micro_steps,
        "scheduled_event_counts": dict(result.scheduled_event_counts),
    }


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
    rank: int = 0,
) -> dict[str, Any]:
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
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
    rank: int = 0,
) -> dict[str, Any] | None:
    if config.data.eval is None:
        return None
    if _explicit_eval_reuses_train_dataset(config):
        return None
    return _resolve_or_build_pack_cache(
        config,
        components,
        vocab_groups,
        repo_root=repo_root,
        dataset=config.data.eval,
        split="eval.forward",
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
    rank: int = 0,
    build_micro_steps: Callable[[int], Sequence[SupervisedMicroStep]],
    materialization_workers: int | None = None,
) -> dict[str, Any]:
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
    cache_root = Path(
        os.environ.get(
            "COORDEXP_SWIFT_PACK_CACHE_ROOT",
            str(repo_root / ".cache" / "coordexp_swift" / "packing"),
        )
    )
    resolved_materialization_workers = _resolve_pack_cache_materialization_workers(
        materialization_workers
    )
    materialization = build_packing_cache_materialization(
        workers=resolved_materialization_workers,
        strategy=PACKING_CACHE_MATERIALIZATION_STRATEGY,
    )
    cache_dir = cache_dir_for_fingerprint(cache_root, fingerprint)
    cache_complete_before = cache_is_complete(cache_dir, fingerprint=fingerprint)
    if rank == 0 and not cache_complete_before:
        micro_steps = tuple(build_micro_steps(resolved_materialization_workers))
        write_micro_step_cache(
            cache_dir,
            micro_steps,
            fingerprint=fingerprint,
            determinants=determinants,
            materialization=materialization,
            augmentation=_augmentation_receipt_from_micro_steps(micro_steps),
        )
    _wait_for_pack_cache(cache_dir, fingerprint=fingerprint)
    manifest = load_cache_manifest(cache_dir)
    cache_manifest_path = manifest_path(cache_dir)
    return {
        "cache_dir": cache_dir,
        "fingerprint": fingerprint,
        "micro_step_count": int(manifest["micro_step_count"]),
        "chunk_count": len(manifest["chunks"]),
        "chunk_size": int(manifest["chunk_size"]),
        "status": manifest["status"],
        "build_status": (
            "hit"
            if cache_complete_before
            else "built"
            if rank == 0
            else "waited"
        ),
        "manifest_path": cache_manifest_path,
        "manifest_sha256": _file_sha256(cache_manifest_path),
        "determinants_sha256": _sha256_json(manifest["determinants"]),
        "chunk_sha256s": [str(chunk["sha256"]) for chunk in manifest["chunks"]],
        "determinants": determinants,
        "materialization": manifest.get("materialization"),
        "augmentation": manifest.get("augmentation"),
    }


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


def _wait_for_pack_cache(
    cache_dir: Path,
    *,
    fingerprint: str,
    timeout_seconds: float = 7200.0,
    poll_seconds: float = 5.0,
) -> None:
    import time

    deadline = time.monotonic() + timeout_seconds
    path = manifest_path(cache_dir)
    while time.monotonic() < deadline:
        if cache_is_complete(cache_dir, fingerprint=fingerprint):
            return
        time.sleep(poll_seconds)
    raise RuntimeContractError(
        "timed out waiting for train packing cache",
        code="training.pack_cache_timeout",
        context={
            "cache_dir": str(cache_dir),
            "manifest_path": str(path),
            "fingerprint": fingerprint,
            "timeout_seconds": timeout_seconds,
        },
    )


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
    adapter_receipt: Any,
    special_token_result: Any,
    trainable_surface: Any,
    processor_identity: dict[str, Any],
    template_identity: dict[str, Any],
    resolved_config_fingerprint: str,
    schedule: ResolvedStepSchedule,
    base_model_path: Path,
    base_config_sha256: str,
    tokenizer_sha256: str,
    best_eval_metrics: BestEvalMetricStore | None = None,
) -> Any:
    def handle(event: ScheduledTrainerEvent) -> None:
        if runtime is not None and not bool(getattr(runtime, "is_main_process", True)):
            return
        step_artifact = event.step_result.to_artifact_dict()
        checkpoint_writer.write_checkpoint(
            planned_step_id=event.scheduled_event.planned_step_id,
            model=_unwrap_checkpoint_model(model, runtime=runtime),
            adapter_receipt=adapter_receipt,
            special_token_result=special_token_result,
            trainable_surface=trainable_surface,
            processor_identity=processor_identity,
            template_identity=template_identity,
            resolved_config_fingerprint=resolved_config_fingerprint,
            schedule_identity={
                "resolved_max_steps": schedule.resolved_max_steps,
                "runtime_batch": schedule.runtime_batch.to_artifact_dict(),
                "events": {
                    name: [
                        item.to_artifact_dict()
                        for item in event_list
                    ]
                    for name, event_list in sorted(schedule.events.items())
                },
            },
            metric_status={
                "finite_status": event.step_result.finite_status,
                "warning_status": "none",
                "loss_bundle": step_artifact.get("loss_bundle", {}),
                "scheduler": step_artifact.get("scheduler", {}),
            },
            optimizer_update_status=event.step_result.optimizer_update_status,
            trigger_reasons=event.scheduled_event.trigger_reasons,
            is_final=(
                event.scheduled_event.planned_step_id == schedule.resolved_max_steps
            ),
            best_metric_event=(
                None
                if best_eval_metrics is None
                else best_eval_metrics.best_for_step(event.scheduled_event.planned_step_id)
            ),
            base_model_path=base_model_path,
            base_config_sha256=base_config_sha256,
            tokenizer_sha256=tokenizer_sha256,
        )

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
    manager: RunArtifactManager,
    eval_source: dict[str, Any] | None,
    best_eval_metrics: BestEvalMetricStore | None = None,
) -> Any:
    def handle(event: ScheduledTrainerEvent) -> None:
        result = ForwardEvalRunner(
            model=model,
            micro_step_stream=iter(eval_micro_steps),
            loss_runner=loss_runner,
            artifact_manager=manager,
            eval_source=eval_source,
            runtime=runtime,
        ).run(
            planned_step_id=event.scheduled_event.planned_step_id,
            trigger_reasons=event.scheduled_event.trigger_reasons,
            optimizer_update_status=event.step_result.optimizer_update_status,
            finite_status=event.step_result.finite_status,
            warning_status="none",
        )
        if best_eval_metrics is not None:
            for metric_event in result.metric_events:
                best_eval_metrics.record(metric_event)

    return handle


def _final_handler() -> Any:
    def handle(event: ScheduledTrainerEvent) -> None:
        del event

    return handle


def _pack_plan_artifact(
    micro_steps: Sequence[SupervisedMicroStep],
    *,
    schedule: ResolvedStepSchedule,
    cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    packs_per_epoch = (
        len(micro_steps)
        if cache is None
        else int(cache["micro_step_count"])
    )
    return {
        "packs_per_epoch": packs_per_epoch,
        "cache": None
        if cache is None
        else {
            "cache_dir": str(cache["cache_dir"]),
            "fingerprint": cache["fingerprint"],
            "global_micro_step_count": cache["micro_step_count"],
            "chunk_count": cache["chunk_count"],
            "chunk_size": cache["chunk_size"],
            "status": cache["status"],
            "build_status": cache.get("build_status"),
            "manifest_path": str(cache["manifest_path"]),
            "manifest_sha256": cache["manifest_sha256"],
            "determinants_sha256": cache["determinants_sha256"],
            "chunk_sha256s": list(cache["chunk_sha256s"]),
            "materialization": cache.get("materialization"),
            "augmentation": cache.get("augmentation"),
        },
        "actual_pack_presentations": schedule.actual_pack_presentations,
        "tail_fill_pack_count": schedule.tail_fill_pack_count,
        "rank_local_micro_step_count": len(micro_steps),
        "rank_local_micro_step_preview": [
            {
                "metadata": dict(micro_step.metadata or {}),
                "pack": _artifact_dict(micro_step.pack),
            }
            for micro_step in micro_steps[:8]
        ],
    }


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


def _explicit_eval_reuses_train_dataset(config: Any) -> bool:
    if config.data.eval is None:
        return False
    return (
        Path(config.data.eval.path).resolve() == Path(config.data.train.path).resolve()
        and config.data.eval.sample_limit == config.data.train.sample_limit
    )


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


def _build_rank_report_gatherer(world_size: int) -> Any | None:
    if world_size <= 1:
        return None

    def gather(local_report: Any) -> tuple[Any, ...]:
        distributed = torch.distributed
        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeContractError(
                "multi-rank finite gates require initialized torch.distributed",
                code="runtime.distributed_gather_uninitialized",
                context={"world_size": world_size},
            )
        gathered: list[Any | None] = [None for _ in range(distributed.get_world_size())]
        distributed.all_gather_object(gathered, local_report)
        return tuple(item for item in gathered if item is not None)

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



def _artifact_dict(value: Any) -> dict[str, Any]:
    to_artifact_dict = getattr(value, "to_artifact_dict", None)
    if callable(to_artifact_dict):
        artifact = to_artifact_dict()
        if isinstance(artifact, dict):
            return artifact
    if isinstance(value, dict):
        return dict(value)
    return {"repr": repr(value), "type": type(value).__name__}
