#!/usr/bin/env python3
"""Production-shaped live entry for the bounded Human-13 R1/R2 successor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


RECEIPT_SCHEMA_VERSION = "human13_row_contrast_runtime.v1"


class SuccessorLiveError(RuntimeError):
    """Raised before an unbound or incomplete successor mutation."""


@dataclass(frozen=True)
class PreparedSuccessorArm:
    config: Any
    config_path: Path
    config_sha256: str
    manifest: Any
    execution_manifest: Any
    ledger: Any
    model_plan: Any
    payload: Any
    assembly: Any
    run_dir: Path
    vertical_image_id: int | None


def build_successor_schedule(*, pack_count: int, max_updates: int) -> Any:
    if isinstance(pack_count, bool) or pack_count <= 0:
        raise SuccessorLiveError("pack_count must be positive")
    if max_updates not in {1, 2} or isinstance(max_updates, bool):
        raise SuccessorLiveError("successor max_updates must be one or two")
    from src.config.models import RuntimeBatchResolution
    from src.training.schedule import ResolvedStepSchedule, StepScheduleEvent

    checkpoints = tuple(
        StepScheduleEvent(
            planned_step_id=step,
            event="checkpoint",
            trigger_reasons=("explicit_step",),
            source_config_path=None,
            deduped_from=(),
            required=True,
        )
        for step in range(1, max_updates + 1)
    )
    final = StepScheduleEvent(
        planned_step_id=max_updates,
        event="final",
        trigger_reasons=("final",),
        source_config_path=None,
        deduped_from=(),
        required=True,
    )
    presentations = pack_count * max_updates
    return ResolvedStepSchedule(
        resolved_max_steps=max_updates,
        packs_per_epoch=pack_count,
        requested_pack_presentations=presentations,
        actual_pack_presentations=presentations,
        tail_fill_pack_count=0,
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=pack_count,
            resolved_grad_accum_steps=pack_count,
        ),
        events={
            "checkpoint": checkpoints,
            "eval.forward": (),
            "final": (final,),
        },
    )


def project_ledger(ledger: Any, *, image_id: int) -> Any:
    def belongs(owner_id: str) -> bool:
        return str(owner_id).startswith(f"gt:{image_id}:")

    positives = tuple(item for item in ledger.positive_rows if belongs(item.owner_id))
    watches = tuple(item for item in ledger.g_watch_rows if belongs(item.owner_id))
    events = tuple(item for item in ledger.events if item.image_id == image_id)
    if not positives or not watches or not events:
        raise SuccessorLiveError("vertical ledger projection is empty")
    return replace(
        ledger,
        positive_rows=positives,
        g_watch_rows=watches,
        events=events,
    )


def select_training_ledger(ledger: Any, config: Any) -> Any:
    """Apply the frozen low-cost training subset without discarding evidence."""

    if (
        config.duplicate_event_sources != "sealed_manifest_only"
        or config.candidate_alias_policy != "first_trajectory_row_per_owner"
    ):
        raise SuccessorLiveError("unknown successor event/alias selection policy")
    events = tuple(
        replace(
            event,
            candidate_groups=tuple(
                replace(group, rows=(group.rows[0],))
                for group in event.candidate_groups
            ),
        )
        for event in ledger.events
        if event.source_kind == "manifest"
    )
    if not events:
        raise SuccessorLiveError("successor training selection has no events")
    return replace(ledger, events=events)


def prepare_successor_arm(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    vertical_image_id: int | None = None,
) -> PreparedSuccessorArm:
    from scripts.research.build_human13_row_contrast_successor import load_ledger
    from scripts.research.human13_live_model import (
        assemble_human13_live_model,
        build_human13_live_model_plan,
        validate_human13_live_model_plan,
    )
    from scripts.research.human13_row_contrast_live import (
        build_successor_payload,
        materialize_successor_segments,
    )
    from scripts.research.materialize_human13_row_contrast_successor import (
        load_successor_config,
        successor_model_config,
    )
    from scripts.research.train_human13_live_arm import (
        _build_skeletons,
        _build_vocab_groups,
        _load_processor_components,
        _load_sealed_manifest,
        project_vertical_manifest,
    )

    root = Path(repo_root).resolve()
    path = Path(config_path).expanduser().resolve(strict=True)
    config = load_successor_config(path, repo_root=root)
    sealed = _load_sealed_manifest(config.manifest_path)
    ledger = load_ledger(config.ledger_path)
    if (
        sealed.manifest_sha256 != config.manifest_sha256
        or ledger.manifest_sha256 != config.manifest_sha256
    ):
        raise SuccessorLiveError("manifest/config/ledger identity differs")
    ledger = select_training_ledger(ledger, config)
    model_config = successor_model_config(config, repo_root=root)
    model_plan = build_human13_live_model_plan(model_config)
    validate_human13_live_model_plan(model_plan)

    components = _load_processor_components(model_plan)
    skeletons = _build_skeletons(sealed.manifest, components, root)
    execution_manifest = sealed.manifest
    execution_ledger = ledger
    selected_skeletons = skeletons
    if vertical_image_id is not None:
        if vertical_image_id != 14038 or config.arm_id != "R1":
            raise SuccessorLiveError("vertical slice is fixed to R1 image 14038")
        execution_manifest = project_vertical_manifest(
            sealed.manifest, image_id=vertical_image_id
        )
        execution_ledger = project_ledger(ledger, image_id=vertical_image_id)
        selected_skeletons = {vertical_image_id: skeletons[vertical_image_id]}
    materialized = materialize_successor_segments(
        execution_ledger,
        selected_skeletons,
        manifest=execution_manifest,
        global_max_length=config.global_max_length,
    )
    vocab_groups = _build_vocab_groups(components)
    payload = build_successor_payload(
        materialized,
        arm_id=config.arm_id,
        expected_vocab_size=int(components.token_identity.tokenizer_vocab_size),
        vocab_groups=vocab_groups,
        global_max_length=config.global_max_length,
        duplicate_margin=config.duplicate_margin,
        rectangle_margin=config.rectangle_margin,
    )
    assembly = assemble_human13_live_model(
        model_plan,
        pack_count=len(payload.micro_steps),
        repo_root=root,
    )
    payload = _attach_image_processor(payload, assembly.components)
    run_dir = Path(config.output_root) / (
        f"vertical-image-{vertical_image_id}"
        if vertical_image_id is not None
        else "training"
    )
    return PreparedSuccessorArm(
        config=config,
        config_path=path,
        config_sha256=_sha256_file(path),
        manifest=sealed.manifest,
        execution_manifest=execution_manifest,
        ledger=execution_ledger,
        model_plan=model_plan,
        payload=payload,
        assembly=assembly,
        run_dir=run_dir.resolve(),
        vertical_image_id=vertical_image_id,
    )


def train_prepared_successor(
    prepared: PreparedSuccessorArm, *, max_updates: int
) -> Mapping[str, Any]:
    if prepared.vertical_image_id is not None and max_updates != 1:
        raise SuccessorLiveError("vertical slice requires one update")
    if prepared.vertical_image_id is None and max_updates != 2:
        raise SuccessorLiveError("full successor requires two cumulative updates")
    from scripts.research.human13_live_model import (
        build_human13_checkpoint_kwargs,
        build_human13_checkpoint_writer,
        readback_human13_checkpoint,
    )
    from scripts.research.human13_row_contrast_live import (
        SuccessorLossRunner,
        build_gradient_projection_handler,
        successor_loss_context_factory,
    )
    from scripts.research.train_human13_live_arm import (
        _peak_memory_bytes,
        write_immutable_receipt,
    )
    from src.artifacts import RunWriter
    from src.training.supervised_trainer import SupervisedTrainer, _default_qwen_forward

    if prepared.run_dir.exists():
        raise SuccessorLiveError(f"run directory already exists: {prepared.run_dir}")
    plan_path = _ensure_resolved_plan(prepared)
    schedule = build_successor_schedule(
        pack_count=len(prepared.payload.micro_steps), max_updates=max_updates
    )
    stream = tuple(prepared.payload.micro_steps) * max_updates
    timestamp = datetime.now(timezone.utc).isoformat()
    run_writer = RunWriter.initialize(
        run_dir=prepared.run_dir,
        run_id=f"human13-{prepared.config.arm_id.lower()}",
        run_name=f"human13-{prepared.config.arm_id.lower()}",
        artifact_root=prepared.run_dir.parent,
        collision_outcome="created",
        created_at=timestamp,
        config_fingerprint=prepared.config_sha256,
        resolved_config=prepared.config.to_artifact_dict(),
        world_size=1,
        resolved_max_steps=max_updates,
    )
    writer = build_human13_checkpoint_writer(prepared.run_dir)
    checkpoint_kwargs = build_human13_checkpoint_kwargs(prepared.assembly)
    readbacks: list[Mapping[str, Any]] = []
    latest: Any | None = None
    started = time.perf_counter()

    def completed(observation: Any) -> None:
        nonlocal latest
        latest = observation
        if (
            observation.optimizer_update_status != "applied"
            or observation.finite_status != "finite"
        ):
            raise SuccessorLiveError("successor update was not finite and applied")
        run_writer.append_logging_row(
            {
                "split": "train",
                "step": observation.planned_step_id,
                "loss_bundle": dict(observation.loss_bundle_artifact),
                "post_backward": dict(observation.post_backward_artifact),
                "scheduler": dict(observation.scheduler_artifact),
                "optimizer_update_status": observation.optimizer_update_status,
                "finite_status": observation.finite_status,
                "non_finite_fields": [],
            }
        )

    def checkpoint(event: Any, observation: Any) -> None:
        step = int(event.planned_step_id)
        result = writer.write_checkpoint(
            step=step,
            model=prepared.assembly.model,
            run_writer=run_writer,
            is_final=step == max_updates,
            **checkpoint_kwargs,
        )
        evidence = readback_human13_checkpoint(
            result.checkpoint_dir,
            expected_step=step,
            assembly=prepared.assembly,
        )
        readbacks.append(evidence.to_artifact_dict())

    runtime_model = getattr(prepared.assembly.runtime, "model", prepared.assembly.model)
    runner = SuccessorLossRunner(
        arm_id=prepared.config.arm_id,
        denominators=prepared.payload.denominators,
        model=runtime_model,
        score_forward=_default_qwen_forward,
        duplicate_margin=prepared.config.duplicate_margin,
        rectangle_margin=prepared.config.rectangle_margin,
    )
    transform = None
    if prepared.config.arm_id == "R2":
        transform = build_gradient_projection_handler(
            prepared.payload,
            epsilon=prepared.config.projection.epsilon,
            tolerance=prepared.config.projection.tolerance,
        )
    trainer = SupervisedTrainer(
        model=prepared.assembly.model,
        schedule=schedule,
        pack_stream=stream,
        loss_context_factory=successor_loss_context_factory,
        loss_runner=runner,
        runtime=prepared.assembly.runtime,
        on_completed_step=completed,
        post_backward_transform=transform,
        on_checkpoint=checkpoint,
        on_final=lambda _event, _observation: None,
    )
    try:
        result = trainer.run()
        if result.completed_steps != max_updates or len(readbacks) != max_updates:
            raise SuccessorLiveError("successor schedule/checkpoint count differs")
        elapsed = time.perf_counter() - started
        performance = prepared.payload.packed_plan.performance_counters(
            wall_time_seconds=elapsed,
            gpu_seconds=None,
            peak_memory_bytes=_peak_memory_bytes(prepared.assembly),
        ).to_artifact_dict()
        run_writer.finalize(
            status="completed",
            updated_at=timestamp,
            completed_steps=max_updates,
            consumed_packs=len(stream),
            checkpoint_event_count=len(readbacks),
            optimizer_update_status=latest.optimizer_update_status,
            finite_status=latest.finite_status,
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "completed",
            "unit_id": prepared.config.unit_id,
            "arm_id": prepared.config.arm_id,
            "config_path": str(prepared.config_path),
            "config_sha256": prepared.config_sha256,
            "resolved_plan_path": str(plan_path),
            "manifest_sha256": prepared.config.manifest_sha256,
            "ledger_sha256": prepared.config.ledger_sha256,
            "vertical_image_id": prepared.vertical_image_id,
            "applied_update_count": max_updates,
            "consumed_pack_count": len(stream),
            "checkpoint_readbacks": readbacks,
            "performance": performance,
            "model_plan": _artifact(prepared.model_plan),
            "trainable_surface": _artifact(prepared.assembly.trainable_surface_receipt),
            "training_result": _artifact(result),
        }
        write_immutable_receipt(
            prepared.run_dir / "human13-row-contrast-runtime-receipt.json", receipt
        )
        return receipt
    except BaseException as error:
        try:
            run_writer.finalize(
                status="failed",
                updated_at=timestamp,
                completed_steps=0 if latest is None else latest.planned_step_id,
                consumed_packs=0,
                checkpoint_event_count=len(readbacks),
                optimizer_update_status=(
                    None if latest is None else latest.optimizer_update_status
                ),
                finite_status=None if latest is None else latest.finite_status,
                terminal_error=f"{type(error).__name__}: {error}",
            )
        except BaseException:
            pass
        raise


def execute_cli(
    *,
    config_path: str | Path,
    repo_root: str | Path,
    execute: bool,
    authority: bool,
    max_updates: int,
    vertical_image_id: int | None,
) -> Mapping[str, Any]:
    from scripts.research.human13_live_model import (
        build_human13_live_model_plan,
        validate_human13_live_model_plan,
    )
    from scripts.research.materialize_human13_row_contrast_successor import (
        ZERO_ACTIONS,
        load_successor_config,
        successor_model_config,
    )

    config = load_successor_config(config_path, repo_root=repo_root)
    model_plan = build_human13_live_model_plan(
        successor_model_config(config, repo_root=repo_root)
    )
    validation = validate_human13_live_model_plan(model_plan)
    if not execute:
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "mode": "dry_run",
            "execution_ready": False,
            "arm_id": config.arm_id,
            "max_updates": max_updates,
            "vertical_image_id": vertical_image_id,
            "manifest_sha256": config.manifest_sha256,
            "ledger_sha256": config.ledger_sha256,
            "model_plan": _artifact(model_plan),
            "source_validation": _artifact(validation),
            "actions": dict(ZERO_ACTIONS),
        }
    if authority is not True:
        raise SuccessorLiveError("execution requires explicit model/GPU authority")
    prepared = prepare_successor_arm(
        config_path,
        repo_root=repo_root,
        vertical_image_id=vertical_image_id,
    )
    return train_prepared_successor(prepared, max_updates=max_updates)


def _attach_image_processor(payload: Any, components: Any) -> Any:
    from src.training.pipeline import (
        _attach_image_processors_to_micro_steps,
        _qwen_image_processor,
    )

    original = tuple(payload.micro_steps)
    attached = _attach_image_processors_to_micro_steps(
        original, image_processor=_qwen_image_processor(components)
    )
    for before_step, after_step in zip(original, attached, strict=True):
        for before, after in zip(
            before_step.encoded_examples, after_step.encoded_examples, strict=True
        ):
            bindings = getattr(before, "human13_successor_bindings", None)
            if bindings is None:
                raise SuccessorLiveError("image processor lost successor bindings")
            object.__setattr__(after, "human13_successor_bindings", bindings)
    return replace(payload, micro_steps=attached)


def _ensure_resolved_plan(prepared: PreparedSuccessorArm) -> Path:
    path = Path(prepared.config.output_root) / "resolved_plan.json"
    payload = prepared.config.to_artifact_dict()
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise SuccessorLiveError("existing resolved plan bytes differ")
        return path
    path.write_bytes(encoded)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(value: Any) -> dict[str, Any]:
    method = getattr(value, "to_artifact_dict", None)
    if callable(method):
        return dict(method())
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return {"repr": repr(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-updates", type=int, choices=(1, 2), required=True)
    parser.add_argument("--vertical-image-id", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = execute_cli(
        config_path=args.config,
        repo_root=args.repo_root,
        execute=args.execute,
        authority=args.user_model_gpu_authority,
        max_updates=args.max_updates,
        vertical_image_id=args.vertical_image_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PreparedSuccessorArm",
    "SuccessorLiveError",
    "build_successor_schedule",
    "execute_cli",
    "prepare_successor_arm",
    "project_ledger",
    "select_training_ledger",
    "train_prepared_successor",
]
