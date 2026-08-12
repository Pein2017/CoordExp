#!/usr/bin/env python3
"""Production-shaped Human-13 live arm training entry.

The module keeps plan inspection CPU-only and crosses the model/GPU boundary
only from explicit execute mode.  It reuses the accepted Swift trainer,
runtime, checkpoint writer, and experiment-local Human-13 payload adapters.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


RECEIPT_SCHEMA_VERSION = "human13_live_training_receipt.v1"
RUNNER_ENTRY_CONTRACT = "human13_runner_cli.v1"
RUNTIME_ENTRY_CONTRACT = "human13_runtime.v1"
CHECKPOINT_MILESTONES = (1, 2, 4, 8, 16)
ALLOWED_UPDATE_COUNTS = frozenset({1, 16})


class LiveTrainingError(RuntimeError):
    """Raised when the live entry cannot preserve the frozen arm contract."""


@dataclass(frozen=True)
class PreparedHuman13LiveArm:
    """One manifest-bound payload after the sole live assembly boundary."""

    arm_id: str
    resolved_plan: Any
    resolved_plan_path: Path
    resolved_plan_sha256: str
    manifest_path: Path
    manifest_sha256: str
    sealed_manifest: Any
    execution_manifest: Any
    model_plan: Any
    payload: Any
    assembly: Any
    run_dir: Path
    vertical_image_id: int | None


def build_training_schedule(*, pack_count: int, max_updates: int) -> Any:
    """Build the frozen cumulative screen or one-step vertical schedule."""

    _require_pack_count(pack_count)
    _require_update_count(max_updates)
    if max_updates == 16:
        from scripts.research.human13_live_model import (
            build_human13_update_schedule,
        )

        return build_human13_update_schedule(pack_count=pack_count)
    from src.config.models import RuntimeBatchResolution
    from src.training.schedule import ResolvedStepSchedule, StepScheduleEvent

    checkpoint_steps = tuple(
        step for step in CHECKPOINT_MILESTONES if step <= max_updates
    )
    checkpoint_events = tuple(
        StepScheduleEvent(
            planned_step_id=step,
            event="checkpoint",
            trigger_reasons=("explicit_step",),
            source_config_path=None,
            deduped_from=(),
            required=step == max_updates,
        )
        for step in checkpoint_steps
    )
    final_event = StepScheduleEvent(
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
            "checkpoint": checkpoint_events,
            "eval.forward": (),
            "final": (final_event,),
        },
    )


def repeat_panel_micro_steps(
    micro_steps: Sequence[Any], *, max_updates: int
) -> tuple[Any, ...]:
    """Repeat complete physical packs without changing within-step order."""

    checked = tuple(micro_steps)
    if not checked:
        raise LiveTrainingError("Human-13 training requires at least one micro-step")
    _require_update_count(max_updates)
    return checked * max_updates


def project_vertical_manifest(manifest: Any, *, image_id: int) -> Any:
    """Derive one image from a sealed parent while using local loss counts."""

    if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id <= 0:
        raise LiveTrainingError("vertical image_id must be a positive integer")
    images = tuple(
        image for image in getattr(manifest, "images", ()) if image.image_id == image_id
    )
    if len(images) != 1:
        raise LiveTrainingError(
            f"vertical image {image_id} does not resolve exactly once"
        )
    if not is_dataclass(manifest):
        raise LiveTrainingError("vertical projection requires a typed manifest")
    image = images[0]
    denominator_type = type(manifest.denominators)
    denominators = denominator_type(
        panel_image_count=1,
        target_image_count=int(bool(image.selected_rows)),
        target_owner_count=len(image.selected_rows),
        replay_image_count=int(bool(image.replay_row_ids)),
        replay_owner_count=len(image.replay_row_ids),
        duplicate_image_count=int(bool(image.duplicate_events)),
        duplicate_event_count=len(image.duplicate_events),
    )
    panel = replace(
        manifest.binding.panel,
        owner_count=len(image.owners),
        images=_project_panel_images(manifest.binding.panel, image_id=image_id),
    )
    binding = replace(manifest.binding, panel=panel)
    return replace(
        manifest,
        binding=binding,
        images=images,
        denominators=denominators,
        # This is a runtime-only derivative of the separately sealed parent.
        # Runner-local admission below never presents it as a new sealed panel.
        full_panel=False,
    )


def train_prepared_arm(
    prepared: PreparedHuman13LiveArm,
    *,
    max_updates: int,
    trainer_type: Any | None = None,
    updated_at: str | None = None,
) -> Any:
    """Run one cumulative model/optimizer stream and publish milestone payloads."""

    if not isinstance(prepared, PreparedHuman13LiveArm):
        raise LiveTrainingError("train_prepared_arm requires one typed preparation")
    _require_update_count(max_updates)
    if prepared.vertical_image_id is not None and max_updates != 1:
        raise LiveTrainingError("vertical slice requires exactly one optimizer update")
    micro_steps = tuple(prepared.payload.micro_steps)
    _require_pack_count(len(micro_steps))
    _validate_prepared_payload(prepared)
    schedule = build_training_schedule(
        pack_count=len(micro_steps), max_updates=max_updates
    )
    pack_stream = repeat_panel_micro_steps(micro_steps, max_updates=max_updates)
    timestamp = updated_at or _utc_now()
    run_writer = _initialize_run_writer(
        prepared,
        max_updates=max_updates,
        created_at=timestamp,
    )
    checkpoint_writer = _build_checkpoint_writer(prepared.run_dir)
    checkpoint_kwargs = _checkpoint_kwargs(prepared.assembly)
    runtime = prepared.assembly.runtime
    initial_optimizer_steps = int(getattr(runtime, "optimizer_step_count", 0))
    checkpoint_readbacks: list[dict[str, Any]] = []
    checkpoint_event_count = 0
    completed_steps = 0
    consumed_packs = 0
    latest: Any | None = None
    started = time.perf_counter()

    def on_completed_step(observation: Any) -> None:
        nonlocal completed_steps, consumed_packs, latest
        if (
            observation.optimizer_update_status != "applied"
            or observation.finite_status != "finite"
        ):
            raise LiveTrainingError(
                f"step {observation.planned_step_id} was not one finite applied update"
            )
        completed_steps = int(observation.planned_step_id)
        consumed_packs += int(observation.micro_step_count)
        latest = observation
        run_writer.append_logging_row(
            {
                "split": "train",
                "step": completed_steps,
                "non_finite_fields": [],
                "pack_count": int(observation.micro_step_count),
                "optimizer_update_status": observation.optimizer_update_status,
                "finite_status": observation.finite_status,
                "loss_bundle": dict(observation.loss_bundle_artifact),
                "scheduler": dict(observation.scheduler_artifact),
                "post_backward": dict(observation.post_backward_artifact),
            }
        )

    def on_checkpoint(event: Any, observation: Any) -> None:
        nonlocal checkpoint_event_count
        step = int(event.planned_step_id)
        if step != int(observation.planned_step_id):
            raise LiveTrainingError("checkpoint event/observation step mismatch")
        result = checkpoint_writer.write_checkpoint(
            step=step,
            model=prepared.assembly.model,
            run_writer=run_writer,
            is_final=step == max_updates,
            **checkpoint_kwargs,
        )
        readback = _readback_checkpoint(
            result.checkpoint_dir,
            expected_step=step,
            assembly=prepared.assembly,
        )
        checkpoint_readbacks.append(dict(readback.to_artifact_dict()))
        checkpoint_event_count += 1

    def on_final(event: Any, observation: Any) -> None:
        if (
            int(event.planned_step_id) != max_updates
            or int(observation.planned_step_id) != max_updates
        ):
            raise LiveTrainingError("final event does not close the requested schedule")

    if trainer_type is None:
        from src.training.supervised_trainer import SupervisedTrainer

        trainer_type = SupervisedTrainer
    try:
        trainer = trainer_type(
            model=prepared.assembly.model,
            schedule=schedule,
            pack_stream=pack_stream,
            loss_context_factory=_loss_context_factory(),
            loss_runner=_loss_runner(prepared.payload.execution_plan),
            runtime=runtime,
            on_completed_step=on_completed_step,
            on_checkpoint=on_checkpoint,
            on_final=on_final,
        )
        result = trainer.run()
        applied_updates = (
            int(getattr(runtime, "optimizer_step_count", 0)) - initial_optimizer_steps
        )
        if (
            applied_updates != max_updates
            or int(result.completed_steps) != max_updates
            or int(result.consumed_micro_steps) != len(pack_stream)
        ):
            raise LiveTrainingError(
                "trainer did not consume the exact cumulative Human-13 schedule"
            )
        expected_checkpoints = tuple(
            step for step in CHECKPOINT_MILESTONES if step <= max_updates
        )
        if tuple(item["step"] for item in checkpoint_readbacks) != expected_checkpoints:
            raise LiveTrainingError(
                "checkpoint readbacks do not cover exact milestones"
            )
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
            consumed_packs=len(pack_stream),
            checkpoint_event_count=checkpoint_event_count,
            optimizer_update_status=(
                None if latest is None else latest.optimizer_update_status
            ),
            finite_status=None if latest is None else latest.finite_status,
        )
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "runner_entry_contract": RUNNER_ENTRY_CONTRACT,
            "runtime_entry_contract": RUNTIME_ENTRY_CONTRACT,
            "execution_ready": True,
            "status": "completed",
            "arm_id": prepared.arm_id,
            "manifest_sha256": prepared.manifest_sha256,
            "parent_manifest_full_panel": True,
            "runtime_projection": (
                None
                if prepared.vertical_image_id is None
                else {
                    "kind": "one_image_vertical_slice",
                    "image_id": prepared.vertical_image_id,
                    "full_panel": False,
                }
            ),
            "resolved_arm_plan_sha256": prepared.resolved_plan_sha256,
            "run_dir": str(prepared.run_dir),
            "applied_update_count": applied_updates,
            "consumed_pack_count": len(pack_stream),
            "checkpoint_readbacks": checkpoint_readbacks,
            "performance": performance,
            "model_plan": _artifact(prepared.model_plan),
            "source_validation": _artifact(prepared.assembly.validation),
            "trainable_surface": _artifact(prepared.assembly.trainable_surface_receipt),
            "memory_savers": _artifact(prepared.assembly.memory_saver_receipt),
            "training_result": _artifact(result),
        }
        receipt["performance"].update(
            {
                "panel_exposure_count": max_updates,
                "total_packed_tokens": int(performance["packed_tokens"]) * max_updates,
                "total_logical_tokens": int(performance["logical_tokens"])
                * max_updates,
            }
        )
        write_immutable_receipt(
            prepared.run_dir / "human13-runtime-receipt.json", receipt
        )
        return result
    except BaseException as error:
        try:
            run_writer.finalize(
                status="failed",
                updated_at=timestamp,
                completed_steps=completed_steps,
                consumed_packs=consumed_packs,
                checkpoint_event_count=checkpoint_event_count,
                optimizer_update_status=(
                    None if latest is None else latest.optimizer_update_status
                ),
                finite_status=None if latest is None else latest.finite_status,
                terminal_error=f"{type(error).__name__}: {error}",
            )
        except BaseException:
            pass
        raise


def prepare_live_arm(
    *,
    resolved_plan_path: str | Path,
    manifest_path: str | Path,
    repo_root: str | Path,
    arm_config_path: str | Path | None = None,
    vertical_image_id: int | None = None,
) -> PreparedHuman13LiveArm:
    """Compose exact processor segments before one fresh live model/runtime."""

    root = Path(repo_root).expanduser().resolve()
    plan_path = Path(resolved_plan_path).expanduser().resolve(strict=True)
    canonical_manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    resolved = _validate_resolved_plan(plan_path, canonical_manifest_path)
    if resolved.updates is not True:
        raise LiveTrainingError("Frozen Source is not a live training arm")
    if vertical_image_id is not None and (
        resolved.arm_id != "A1" or vertical_image_id != 14038
    ):
        raise LiveTrainingError(
            "the authorized vertical slice is exactly A1 on image 14038"
        )
    config_path = _resolve_arm_config(resolved, arm_config_path, root)
    model_plan = _build_model_plan(config_path)
    if model_plan.arm_id != resolved.arm_id:
        raise LiveTrainingError("arm config and resolved plan arm identities differ")
    _validate_model_plan(model_plan)
    sealed = _load_sealed_manifest(canonical_manifest_path)
    if sealed.manifest_sha256 != resolved.manifest_sha256:
        raise LiveTrainingError("resolved plan and sealed manifest digests differ")

    # The processor-only pass is needed to know exact pack_count before
    # TrainRuntime is assembled with its planned-step accumulation contract.
    processor_components = _load_processor_components(model_plan)
    skeletons = _build_skeletons(sealed.manifest, processor_components, root)
    execution_manifest = sealed.manifest
    selected_skeletons = skeletons
    if vertical_image_id is not None:
        execution_manifest = project_vertical_manifest(
            sealed.manifest, image_id=vertical_image_id
        )
        selected_skeletons = {vertical_image_id: skeletons[vertical_image_id]}
    materialized = _materialize_segments(execution_manifest, selected_skeletons)
    payload = _build_live_payload(
        sealed_parent=sealed,
        execution_manifest=execution_manifest,
        materialized_segments=materialized,
        arm_id=resolved.arm_id,
        processor_components=processor_components,
        resolved_plan_raw=resolved.raw,
        vertical=vertical_image_id is not None,
    )
    pack_count = len(payload.micro_steps)
    _require_pack_count(pack_count)
    assembly = _assemble_live_model(model_plan, pack_count, root)
    payload = _attach_live_image_processor(payload, assembly.components)
    run_dir = Path(resolved.output_root) / (
        f"vertical-slice-image-{vertical_image_id}"
        if vertical_image_id is not None
        else "training"
    )
    return PreparedHuman13LiveArm(
        arm_id=resolved.arm_id,
        resolved_plan=resolved,
        resolved_plan_path=plan_path,
        resolved_plan_sha256=_sha256_file(plan_path),
        manifest_path=canonical_manifest_path,
        manifest_sha256=sealed.manifest_sha256,
        sealed_manifest=sealed,
        execution_manifest=execution_manifest,
        model_plan=model_plan,
        payload=payload,
        assembly=assembly,
        run_dir=run_dir.resolve(),
        vertical_image_id=vertical_image_id,
    )


def execute_cli(
    *,
    resolved_plan_path: str | Path,
    manifest_path: str | Path,
    repo_root: str | Path,
    execute: bool,
    authority: bool,
    vertical_slice: bool,
    image_id: int | None,
    max_updates: int,
    arm_config_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Fail-closed CLI boundary; only explicit execute plus authority trains."""

    if vertical_slice != (image_id is not None):
        raise LiveTrainingError(
            "--vertical-slice and --image-id must be provided together"
        )
    if vertical_slice and (image_id != 14038 or max_updates != 1):
        raise LiveTrainingError(
            "vertical slice is fixed to --image-id 14038 --max-updates 1"
        )
    if not vertical_slice and max_updates != 16:
        raise LiveTrainingError("full-panel training requires --max-updates 16")
    if execute and authority is not True:
        raise LiveTrainingError("execute mode requires explicit model/GPU authority")
    if not execute:
        resolved = _validate_resolved_plan(resolved_plan_path, manifest_path)
        config_path = _resolve_arm_config(
            resolved, arm_config_path, Path(repo_root).resolve()
        )
        model_plan = _build_model_plan(config_path)
        validation = _validate_model_plan(model_plan)
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "runner_entry_contract": RUNNER_ENTRY_CONTRACT,
            "runtime_entry_contract": RUNTIME_ENTRY_CONTRACT,
            "mode": "dry_run",
            "execution_ready": False,
            "arm_id": resolved.arm_id,
            "manifest_sha256": resolved.manifest_sha256,
            "max_updates": max_updates,
            "vertical_image_id": image_id,
            "model_plan": _artifact(model_plan),
            "source_validation": _artifact(validation),
            "actions": {
                "model_loads": 0,
                "forwards": 0,
                "backwards": 0,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "gpu_allocations": 0,
            },
        }
    prepared = prepare_live_arm(
        resolved_plan_path=resolved_plan_path,
        manifest_path=manifest_path,
        repo_root=repo_root,
        arm_config_path=arm_config_path,
        vertical_image_id=image_id,
    )
    train_prepared_arm(prepared, max_updates=max_updates)
    receipt_path = prepared.run_dir / "human13-runtime-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("runtime_entry_contract") != RUNTIME_ENTRY_CONTRACT
        or receipt.get("execution_ready") is not True
        or receipt.get("manifest_sha256") != prepared.manifest_sha256
        or receipt.get("arm_id") != prepared.arm_id
    ):
        raise LiveTrainingError("runtime receipt does not bind the executed arm")
    return receipt


def write_immutable_receipt(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish one canonical receipt exactly once."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LiveTrainingError(f"runtime receipt already exists: {output}") from exc
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return output


def materialize_resolved_plan(
    plans_receipt: Mapping[str, Any] | str | Path,
    *,
    arm_id: str,
) -> Path:
    """Publish one exact arm plan from the CPU materializer receipt.

    ``resolved_plan_path`` is transport metadata owned by the receipt, not part
    of the content-bound plan consumed by ``validate_resolved_plan``.
    """

    if isinstance(plans_receipt, Mapping):
        receipt = plans_receipt
    else:
        try:
            receipt = json.loads(
                Path(plans_receipt)
                .expanduser()
                .resolve(strict=True)
                .read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LiveTrainingError(
                f"cannot load materialized plans receipt: {exc}"
            ) from exc
    if receipt.get("schema_version") != "human13_materialized_plans.v1":
        raise LiveTrainingError("materialized plans receipt schema differs")
    plans = receipt.get("plans")
    if not isinstance(plans, list):
        raise LiveTrainingError("materialized plans receipt has no plans")
    matches = tuple(
        plan
        for plan in plans
        if isinstance(plan, Mapping) and plan.get("arm_id") == arm_id
    )
    if len(matches) != 1:
        raise LiveTrainingError(f"arm {arm_id} does not resolve exactly once")
    transported = dict(matches[0])
    destination_value = transported.pop("resolved_plan_path", None)
    output_root = transported.get("output_root")
    if not isinstance(destination_value, str) or not isinstance(output_root, str):
        raise LiveTrainingError("arm plan has no resolved plan/output path")
    destination = Path(destination_value).expanduser()
    root = Path(output_root).expanduser()
    if not destination.is_absolute() or not root.is_absolute():
        raise LiveTrainingError("resolved plan and output_root must be absolute")
    if destination != root / "resolved_plan.json":
        raise LiveTrainingError(
            "resolved_plan_path must be output_root/resolved_plan.json"
        )
    return write_immutable_receipt(destination, transported)


def _require_pack_count(pack_count: int) -> None:
    if (
        isinstance(pack_count, bool)
        or not isinstance(pack_count, int)
        or pack_count <= 0
    ):
        raise LiveTrainingError("pack_count must be a positive integer")


def _require_update_count(max_updates: int) -> None:
    if max_updates not in ALLOWED_UPDATE_COUNTS or isinstance(max_updates, bool):
        raise LiveTrainingError("max_updates must be exactly 1 or 16")


def _project_panel_images(panel: Any, *, image_id: int) -> tuple[Any, ...]:
    identities = getattr(panel, "images", ())
    matched = tuple(
        item
        for item in identities
        if getattr(item, "image_id", None) == image_id or item == f"image-{image_id}"
    )
    if identities and not matched:
        raise LiveTrainingError(
            f"panel identity does not contain vertical image {image_id}"
        )
    return matched


def _validate_resolved_plan(path: str | Path, manifest_path: str | Path) -> Any:
    from scripts.research.execute_human13_k_union import validate_resolved_plan

    return validate_resolved_plan(path, manifest_path)


def _resolve_arm_config(
    resolved_plan: Any,
    arm_config_path: str | Path | None,
    repo_root: Path,
) -> Path:
    from scripts.research.materialize_human13_k_union_configs import (
        CONFIG_ROOT,
        load_arm_config,
    )

    if arm_config_path is not None:
        path = Path(arm_config_path)
        path = path if path.is_absolute() else repo_root / path
        path = path.resolve(strict=True)
        if load_arm_config(path).arm_id != resolved_plan.arm_id:
            raise LiveTrainingError("explicit arm config differs from resolved plan")
        return path
    root = CONFIG_ROOT if CONFIG_ROOT.is_absolute() else repo_root / CONFIG_ROOT
    matches = tuple(
        path.resolve()
        for path in sorted(root.glob("*.yaml"))
        if load_arm_config(path).arm_id == resolved_plan.arm_id
    )
    if len(matches) != 1:
        raise LiveTrainingError(
            f"arm {resolved_plan.arm_id} does not resolve to exactly one config"
        )
    return matches[0]


def _build_model_plan(path: str | Path) -> Any:
    from scripts.research.human13_live_model import build_human13_live_model_plan

    return build_human13_live_model_plan(path)


def _validate_model_plan(plan: Any) -> Any:
    from scripts.research.human13_live_model import validate_human13_live_model_plan

    return validate_human13_live_model_plan(plan)


def _load_sealed_manifest(path: str | Path) -> Any:
    from scripts.research.run_human13_k_union_overfit import (
        load_sealed_training_manifest,
    )

    return load_sealed_training_manifest(path)


def _load_processor_components(model_plan: Any) -> Any:
    from src.qwen.runtime_loading import (
        QwenLoadOptions,
        load_qwen_components_from_options,
    )

    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=model_plan.source.base_model_path,
            dtype=model_plan.mixed_precision,
            attn_implementation=model_plan.attn_implementation,
            patch_embed_linearization=model_plan.patch_embed_linearization,
            load_model=False,
        )
    )


def _build_skeletons(manifest: Any, components: Any, repo_root: Path) -> dict[int, Any]:
    from scripts.research.human13_live_model import build_human13_processor_skeletons

    return build_human13_processor_skeletons(
        manifest,
        components,
        repo_root=repo_root,
    )


def _materialize_segments(manifest: Any, skeletons: Mapping[int, Any]) -> Any:
    from scripts.research.human13_live_segments import materialize_segments

    return materialize_segments(manifest, skeletons)


def _build_vocab_groups(components: Any) -> Any:
    from src.losses import build_token_vocabulary_groups

    return build_token_vocabulary_groups(
        components.token_identity,
        tokenizer=components.tokenizer,
    )


def _build_live_payload(
    *,
    sealed_parent: Any,
    execution_manifest: Any,
    materialized_segments: Any,
    arm_id: str,
    processor_components: Any,
    resolved_plan_raw: Mapping[str, Any],
    vertical: bool,
) -> Any:
    from scripts.research import human13_live_payload as live_payload

    vocab_groups = _build_vocab_groups(processor_components)
    expected_vocab_size = int(processor_components.token_identity.tokenizer_vocab_size)
    a6 = _typed_a6_binding(resolved_plan_raw.get("a6_donor_binding"))
    a8 = _typed_a8_binding(resolved_plan_raw.get("a8_census_binding"))
    if vertical:
        if arm_id != "A1" or a6 is not None or a8 is not None:
            raise LiveTrainingError("vertical runtime projection only admits A1")
        return _build_vertical_a1_payload(
            sealed_parent=sealed_parent,
            execution_manifest=execution_manifest,
            materialized_segments=materialized_segments,
            expected_vocab_size=expected_vocab_size,
            vocab_groups=vocab_groups,
        )
    return live_payload.build_live_payload(
        sealed_manifest=sealed_parent,
        materialized_segments=materialized_segments,
        arm_id=arm_id,
        expected_vocab_size=expected_vocab_size,
        vocab_groups=vocab_groups,
        a6_donor_binding=a6,
        a8_census_binding=a8,
    )


def _build_vertical_a1_payload(
    *,
    sealed_parent: Any,
    execution_manifest: Any,
    materialized_segments: Any,
    expected_vocab_size: int,
    vocab_groups: Any,
) -> Any:
    """Runner-local admission for the authorized one-image A1 projection."""

    from scripts.research import human13_live_payload as live_payload
    from scripts.research import run_human13_k_union_overfit as runner
    from src.supervision import build_token_sequence_from_packed_supervision

    if getattr(execution_manifest, "full_panel", None) is not False:
        raise LiveTrainingError(
            "vertical runtime projection must remain full_panel=false"
        )
    selected = live_payload._select_segments(
        materialized_segments,
        arm_id="A1",
        global_max_length=runner.GLOBAL_MAX_LENGTH,
    )
    packed_plan = runner.plan_panel_packs(selected)
    sites_by_pack = {
        packed.pack.pack_index: live_payload._sites_for_pack(
            packed,
            arm_id="A1",
            a8_census_binding=None,
        )
        for packed in packed_plan.packs
    }
    token_sequences = {
        packed.pack.pack_index: build_token_sequence_from_packed_supervision(
            packed.pack, ()
        )
        for packed in packed_plan.packs
    }
    contract = runner._arm_contract("A1")
    denominators = runner._manifest_denominators(
        execution_manifest,
        "A1",
        contract.coefficients,
    )
    micro_steps = runner.build_supervised_micro_steps(
        packed_plan,
        denominators=denominators,
        token_sequences=token_sequences,
        vocab_groups=vocab_groups,
        sites_by_pack=sites_by_pack,
        expected_vocab_size=expected_vocab_size,
    )
    projected_sealed = runner.SealedHuman13Manifest(
        manifest=execution_manifest,
        manifest_sha256=sealed_parent.manifest_sha256,
    )
    execution_plan = runner.Human13ExecutionPlan(
        manifest_identity=runner._manifest_identity(projected_sealed),
        arm_id="A1",
        denominators=denominators,
        coefficients=contract.coefficients,
        pack_segments=tuple(
            (packed.pack.pack_index, runner._segment_bindings(packed))
            for packed in packed_plan.packs
        ),
        sites_by_pack=tuple(
            (packed.pack.pack_index, tuple(sites_by_pack[packed.pack.pack_index]))
            for packed in packed_plan.packs
        ),
    )
    runner._validate_sites_against_manifest(
        execution_manifest,
        execution_plan,
        packed_plan=packed_plan,
        contract=contract,
    )
    return live_payload.Human13LivePayload(
        arm_id="A1",
        selected_segments=selected,
        packed_plan=packed_plan,
        sites_by_pack=sites_by_pack,
        token_sequences=token_sequences,
        micro_steps=micro_steps,
        execution_plan=execution_plan,
    )


def _assemble_live_model(model_plan: Any, pack_count: int, repo_root: Path) -> Any:
    from scripts.research.human13_live_model import assemble_human13_live_model

    return assemble_human13_live_model(
        model_plan,
        pack_count=pack_count,
        repo_root=repo_root,
    )


def _attach_live_image_processor(payload: Any, components: Any) -> Any:
    from src.training.pipeline import (
        _attach_image_processors_to_micro_steps,
        _qwen_image_processor,
    )

    attached = _attach_image_processors_to_micro_steps(
        tuple(payload.micro_steps),
        image_processor=_qwen_image_processor(components),
    )
    return replace(payload, micro_steps=attached)


def _typed_manifest_identity(raw: Mapping[str, Any]) -> Any:
    from scripts.research.run_human13_k_union_overfit import Human13ManifestIdentity

    return Human13ManifestIdentity(
        schema_version=str(raw["schema_version"]),
        unit_id=str(raw["unit_id"]),
        panel_sha256=str(raw["panel_sha256"]),
        manifest_sha256=str(raw["manifest_sha256"]),
    )


def _typed_a6_binding(raw: Any) -> Any | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise LiveTrainingError("resolved A6 donor binding must be an object")
    from scripts.research.run_human13_k_union_overfit import (
        Human13A6DonorBinding,
        Human13A6DonorRecord,
    )

    identity = _typed_manifest_identity(dict(raw["manifest_identity"]))
    donors = tuple(
        Human13A6DonorRecord(
            image_id=int(item["image_id"]),
            owner_id=str(item["owner_id"]),
            target_row_id=str(item["target_row_id"]),
            donor_trajectory_id=str(item["donor_trajectory_id"]),
            donor_prefix_token_ids=tuple(
                int(x) for x in item["donor_prefix_token_ids"]
            ),
            donor_prior_row_ids=tuple(str(x) for x in item["donor_prior_row_ids"]),
            h_mid_eligible=bool(item["h_mid_eligible"]),
        )
        for item in raw["donors"]
    )
    return Human13A6DonorBinding(
        schema_version=str(raw["schema_version"]),
        manifest_identity=identity,
        frozen_targets_sha256=str(raw["frozen_targets_sha256"]),
        artifact_sha256=str(raw["artifact_sha256"]),
        applicable=bool(raw["applicable"]),
        donors=donors,
    )


def _typed_a8_binding(raw: Any) -> Any | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise LiveTrainingError("resolved A8 census binding must be an object")
    from scripts.research.run_human13_k_union_overfit import Human13A8CensusBinding

    return Human13A8CensusBinding(
        schema_version=str(raw["schema_version"]),
        census_schema_version=str(raw["census_schema_version"]),
        manifest_identity=_typed_manifest_identity(dict(raw["manifest_identity"])),
        frozen_targets_sha256=str(raw["frozen_targets_sha256"]),
        artifact_sha256=str(raw["artifact_sha256"]),
        applicable=bool(raw["applicable"]),
        required_margin=float(raw["required_margin"]),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_run_writer(
    prepared: PreparedHuman13LiveArm,
    *,
    max_updates: int,
    created_at: str,
) -> Any:
    from src.artifacts import RunWriter

    run_id = f"human13-{prepared.arm_id.lower()}"
    if prepared.vertical_image_id is not None:
        run_id += f"-vertical-{prepared.vertical_image_id}"
    return RunWriter.initialize(
        run_dir=prepared.run_dir,
        run_id=run_id,
        run_name=run_id,
        artifact_root=prepared.run_dir.parent,
        collision_outcome="created",
        created_at=created_at,
        config_fingerprint=prepared.resolved_plan_sha256,
        resolved_config=dict(prepared.resolved_plan.raw),
        world_size=1,
        resolved_max_steps=max_updates,
    )


def _build_checkpoint_writer(run_dir: Path) -> Any:
    from scripts.research.human13_live_model import (
        build_human13_checkpoint_writer,
    )

    return build_human13_checkpoint_writer(run_dir)


def _checkpoint_kwargs(assembly: Any) -> dict[str, Any]:
    from scripts.research.human13_live_model import (
        build_human13_checkpoint_kwargs,
    )

    return build_human13_checkpoint_kwargs(assembly)


def _readback_checkpoint(
    checkpoint_dir: Path, *, expected_step: int, assembly: Any
) -> Any:
    from scripts.research.human13_live_model import readback_human13_checkpoint

    return readback_human13_checkpoint(
        checkpoint_dir,
        expected_step=expected_step,
        assembly=assembly,
    )


def _validate_prepared_payload(prepared: PreparedHuman13LiveArm) -> None:
    from scripts.research import run_human13_k_union_overfit as runner

    sealed = runner.SealedHuman13Manifest(
        manifest=prepared.execution_manifest,
        manifest_sha256=prepared.manifest_sha256,
    )
    runner._validate_execution_payload(
        sealed,
        prepared.payload.execution_plan,
        tuple(prepared.payload.micro_steps),
    )


def _loss_context_factory() -> Any:
    from scripts.research.run_human13_k_union_overfit import (
        human13_loss_context_factory,
    )

    return human13_loss_context_factory


def _loss_runner(execution_plan: Any) -> Any:
    from scripts.research.run_human13_k_union_overfit import Human13PanelLossRunner

    return Human13PanelLossRunner(
        denominators=execution_plan.denominators,
        coefficients=execution_plan.coefficients,
    )


def _peak_memory_bytes(assembly: Any) -> int | None:
    device = getattr(getattr(assembly, "accelerator", None), "device", None)
    if device is None:
        return None
    try:
        import torch

        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(resolved))
    except (RuntimeError, TypeError, ValueError):
        return None


def _artifact(value: Any) -> dict[str, Any]:
    method = getattr(value, "to_artifact_dict", None)
    if callable(method):
        artifact = method()
        if isinstance(artifact, Mapping):
            return dict(artifact)
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return {"repr": repr(value)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    plan_source = parser.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--resolved-plan", type=Path)
    plan_source.add_argument("--plans-receipt", type=Path)
    parser.add_argument("--arm-id")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm-config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-model-gpu-authority", action="store_true")
    parser.add_argument("--vertical-slice", action="store_true")
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--max-updates", type=int, choices=(1, 16), default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plans_receipt is not None:
        if not args.arm_id:
            raise LiveTrainingError("--plans-receipt requires --arm-id")
        resolved_plan = materialize_resolved_plan(
            args.plans_receipt,
            arm_id=args.arm_id,
        )
    else:
        if args.arm_id:
            raise LiveTrainingError("--arm-id is only valid with --plans-receipt")
        resolved_plan = args.resolved_plan
    receipt = execute_cli(
        resolved_plan_path=resolved_plan,
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        execute=args.execute,
        authority=args.user_model_gpu_authority,
        vertical_slice=args.vertical_slice,
        image_id=args.image_id,
        max_updates=args.max_updates,
        arm_config_path=args.arm_config,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_MILESTONES",
    "LiveTrainingError",
    "PreparedHuman13LiveArm",
    "RECEIPT_SCHEMA_VERSION",
    "RUNNER_ENTRY_CONTRACT",
    "RUNTIME_ENTRY_CONTRACT",
    "build_parser",
    "build_training_schedule",
    "execute_cli",
    "main",
    "materialize_resolved_plan",
    "prepare_live_arm",
    "project_vertical_manifest",
    "repeat_panel_micro_steps",
    "train_prepared_arm",
    "write_immutable_receipt",
]
