"""Training pipeline assembly for the V1 supervised smoke."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Any

import torch

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - exercised only in stripped environments.
    Accelerator = None  # type: ignore[assignment]

from src.adapters import (
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
    setup_dora_adapter,
)
from src.artifacts import MetricStreamEvent, RunArtifactManager
from src.artifacts.checkpoints import CheckpointWriter
from src.common.errors import RuntimeContractError
from src.config.loader import load_train_config
from src.config.paths import resolve_run_directory
from src.config.resolve import resolve_qwen_runtime_controls
from src.data import load_raw_examples
from src.eval import ForwardEvalRunner
from src.losses import LossRunner, build_token_vocabulary_groups
from src.optim import (
    build_optimizer_and_scheduler,
    build_optimizer_group_plan,
    build_trainable_surface_receipt,
)
from src.packing import PackedSequence, build_packed_supervision, plan_packed_sequences
from src.qwen import (
    build_default_special_token_selection,
    build_qwen_position_inputs,
    encode_rendered_example,
    load_qwen_components,
)
from src.qwen.special_token_embeddings import (
    load_default_special_token_embedding_source_gate_evidence,
    install_special_token_embedding_deltas,
)
from src.runtime import TrainRuntime
from src.supervision import build_token_sequence_from_packed_supervision
from src.templates import render_example
from src.training.schedule import ResolvedStepSchedule, resolve_planned_step_schedule
from src.training.supervised_trainer import (
    ScheduledTrainerEvent,
    SupervisedMicroStep,
    SupervisedTrainer,
    SupervisedTrainerEvent,
)


TRAIN_SPLIT = "train"
BEST_EVAL_SELECTOR_SPLIT = "eval.forward"
BEST_EVAL_SELECTOR_NAME = "acc_top1"


def build_repeating_micro_step_stream(
    base_micro_steps: Sequence[SupervisedMicroStep],
    schedule: ResolvedStepSchedule,
) -> Iterator[SupervisedMicroStep]:
    if not base_micro_steps:
        raise ValueError("base_micro_steps must contain at least one micro-step")
    total_rank_local_micro_steps = (
        schedule.resolved_max_steps
        * schedule.runtime_batch.resolved_grad_accum_steps
    )

    def iter_repeated() -> Iterator[SupervisedMicroStep]:
        for index in range(total_rank_local_micro_steps):
            yield base_micro_steps[index % len(base_micro_steps)]

    return iter_repeated()


@dataclass
class TrainingArtifactBridge:
    manager: RunArtifactManager
    rank: int
    world_size: int

    def __call__(self, event: SupervisedTrainerEvent) -> None:
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


def run_training_pipeline(config_path: str | Path) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    resolved_config = load_train_config(config_path)
    config = resolved_config.config
    run_directory = _resolve_rank_local_run_directory(config, cwd=repo_root)
    created_at = datetime.now(UTC).isoformat()
    manager = RunArtifactManager.initialize(
        run_directory=run_directory,
        run_id=_run_id(config.run.name, resolved_config.fingerprint),
        created_at=created_at,
        runtime_identity={
            "entrypoint": "python -m src.train",
            "backend": config.runtime.backend,
            "cwd": str(repo_root),
        },
        backend_status=_first_smoke_backend_status(config.runtime.backend),
    )
    manager.write_resolved_config(resolved_config)

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
    train_micro_steps = build_base_micro_steps(config, components, vocab_groups)
    schedule = resolve_planned_step_schedule(
        config,
        packs_per_epoch=len(train_micro_steps),
        world_size=_world_size(),
        source_config_path=str(resolved_config.entry_config_path),
    )
    manager.write_schedule(schedule)
    manager.write_receipt(
        "pack_plan",
        _pack_plan_artifact(
            train_micro_steps,
            schedule=schedule,
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

    device = _runtime_device(_rank())
    accelerator = _build_accelerator(config.runtime, schedule.runtime_batch)
    runtime = TrainRuntime(
        runtime_config=config.runtime,
        runtime_batch=schedule.runtime_batch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=_rank(),
        world_size=_world_size(),
        max_grad_norm=config.training.max_grad_norm,
        accelerator=accelerator,
        rank_report_gatherer=_build_rank_report_gatherer(_world_size()),
    )
    manager.write_receipt(
        "runtime_setup",
        _artifact_dict(runtime.setup_receipt),
        category="runtime",
    )

    eval_micro_steps = (
        train_micro_steps
        if _explicit_eval_reuses_train_dataset(config)
        else _build_micro_steps_for_dataset(
            config,
            components,
            vocab_groups,
            dataset=config.data.eval,
            split="eval.forward",
        )
    )
    checkpoint_writer = CheckpointWriter(manager)
    best_eval_metrics = BestEvalMetricStore()
    trainer = SupervisedTrainer(
        model=model,
        schedule=schedule,
        pack_stream=build_repeating_micro_step_stream(train_micro_steps, schedule),
        loss_runner=loss_runner,
        runtime=runtime,
        event_sink=TrainingArtifactBridge(
            manager,
            rank=runtime.rank,
            world_size=runtime.world_size,
        ),
        scheduled_event_handlers={
            "checkpoint": _checkpoint_handler(
                checkpoint_writer,
                model=runtime.model,
                runtime=runtime,
                adapter_receipt=adapter_result.receipt,
                special_token_result=special_token_result,
                trainable_surface=trainable_surface,
                processor_identity=_artifact_dict(components.processor_identity),
                resolved_config_fingerprint=resolved_config.fingerprint,
                schedule=schedule,
                base_model_path=components.base_model_path,
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
) -> tuple[SupervisedMicroStep, ...]:
    return _build_micro_steps_for_dataset(
        config,
        components,
        vocab_groups,
        dataset=config.data.train,
        split=TRAIN_SPLIT,
    )


def _build_micro_steps_for_dataset(
    config: Any,
    components: Any,
    vocab_groups: Any,
    *,
    dataset: Any,
    split: str,
) -> tuple[SupervisedMicroStep, ...]:
    if dataset is None:
        raise RuntimeContractError(
            "micro-step construction requires an explicit dataset split",
            code="training.dataset_split_missing",
            context={"split": split},
        )
    raw_examples = load_raw_examples(dataset)
    rendered_examples = tuple(
        render_example(
            raw_example,
            config.template,
            object_order_seed=_object_order_seed(config, raw_example.example_id),
        )
        for raw_example in raw_examples
    )
    encoded_examples = tuple(
        encode_rendered_example(
            raw_example,
            rendered,
            components=components,
            processor_config=config.model.processor,
            global_max_length=config.packing.global_max_length,
        )
        for raw_example, rendered in zip(raw_examples, rendered_examples, strict=True)
    )
    packs = plan_packed_sequences(
        encoded_examples,
        global_max_length=config.packing.global_max_length,
    )
    supervision = build_packed_supervision(packs, encoded_examples)
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
            supervision,
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
                },
                expected_vocab_size=components.token_identity.tokenizer_vocab_size,
                fa2_model_dtype=config.training.precision,
                capture_fa2_branch=True,
                require_fa2_branch_proof=True,
            )
        )
    if not micro_steps:
        raise RuntimeContractError(
            "micro-step construction produced no packs",
            code="training.empty_micro_step_plan",
            context={"split": split},
        )
    return tuple(micro_steps)


def enable_training_memory_savers(model: Any) -> dict[str, Any]:
    train_mode_enabled = _enable_train_mode(model)
    use_cache_disabled = _disable_use_cache(model)
    gradient_checkpointing_enabled = _call_first_available(
        model,
        "gradient_checkpointing_enable",
    )
    input_require_grads_enabled = _call_first_available(
        model,
        "enable_input_require_grads",
    )
    return {
        "train_mode_enabled": train_mode_enabled,
        "model_training": _model_training_state(model),
        "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
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
    resolved_config_fingerprint: str,
    schedule: ResolvedStepSchedule,
    base_model_path: Path,
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
        )

    return handle


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
) -> dict[str, Any]:
    return {
        "packs_per_epoch": len(micro_steps),
        "actual_pack_presentations": schedule.actual_pack_presentations,
        "tail_fill_pack_count": schedule.tail_fill_pack_count,
        "micro_steps": [
            {
                "metadata": dict(micro_step.metadata or {}),
                "pack": _artifact_dict(micro_step.pack),
                "position_inputs": _artifact_dict(micro_step.position_inputs),
                "token_sequence": _artifact_dict(micro_step.token_sequence),
            }
            for micro_step in micro_steps
        ],
    }


def _resolve_rank_local_run_directory(config: Any, *, cwd: Path) -> Any:
    rank = _rank()
    world_size = _world_size()
    timestamp = None
    if world_size > 1 and rank > 0:
        timestamp = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-rank{rank}"
    run_directory = resolve_run_directory(config, cwd=cwd, timestamp=timestamp)
    if world_size <= 1 or rank == 0:
        return run_directory
    rank_suffix = f"rank{rank}"
    if run_directory.run_dir.name.endswith(f"-{rank_suffix}"):
        return run_directory
    rank_run_dir = run_directory.run_dir.with_name(
        f"{run_directory.run_dir.name}-{rank_suffix}"
    )
    if rank_run_dir.exists():
        if config.run.collision_policy == "fail":
            raise RuntimeContractError(
                "rank-local run output directory already exists",
                code="runtime.rank_run_dir_exists",
                context={
                    "rank": rank,
                    "world_size": world_size,
                    "run_dir": str(rank_run_dir),
                },
            )
        rank_run_dir = run_directory.run_dir.with_name(
            f"{run_directory.run_dir.name}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{rank_suffix}"
        )
    return replace(run_directory, run_dir=rank_run_dir.resolve())


def _loss_plan_artifact(config: Any, vocab_groups_artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalizer": config.losses.normalizer,
        "objective_dtype": "float32_selected_logits",
        "protected": config.losses.protected.model_dump(mode="json"),
        "term_order": ["base_ce", "token_type_gate"],
        "vocabulary_groups": vocab_groups_artifact,
        "finite_policy": {
            "pre_backward_scalar_gate": "all_rank_consensus",
            "post_backward_gradient_gate": "all_rank_consensus",
        },
        "metric_definitions": {
            "top_level": ["acc_top1", "acc_top5"],
            "weighted_losses": ["loss/base_ce", "loss/token_type_gate", "loss/total"],
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


def _build_accelerator(
    runtime_config: Any,
    runtime_batch: Any,
) -> Any | None:
    if runtime_config.backend != "accelerate":
        return None
    if Accelerator is None:
        raise RuntimeContractError(
            "accelerate backend requires the accelerate package",
            code="runtime.accelerate_unavailable",
        )
    accelerate_config = runtime_config.accelerate
    kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": runtime_batch.resolved_grad_accum_steps,
    }
    mixed_precision = (
        None
        if accelerate_config is None
        else accelerate_config.mixed_precision
    )
    if mixed_precision is not None:
        kwargs["mixed_precision"] = mixed_precision
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


def _runtime_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        device_count = max(1, torch.cuda.device_count())
        return torch.device(f"cuda:{rank % device_count}")
    return torch.device("cpu")


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _run_id(run_name: str, fingerprint: str) -> str:
    return f"{run_name}-{fingerprint[:12]}"


def _first_smoke_backend_status(runtime_backend: str) -> dict[str, list[str]]:
    return {
        "active": [runtime_backend],
        "single": ["active"] if runtime_backend == "single" else ["schema_accepted"],
        "accelerate": ["schema_accepted"],
        "deepspeed": ["schema_accepted", "conflict_validation_implemented"],
    }


def _artifact_dict(value: Any) -> dict[str, Any]:
    to_artifact_dict = getattr(value, "to_artifact_dict", None)
    if callable(to_artifact_dict):
        artifact = to_artifact_dict()
        if isinstance(artifact, dict):
            return artifact
    if isinstance(value, dict):
        return dict(value)
    return {"repr": repr(value), "type": type(value).__name__}
