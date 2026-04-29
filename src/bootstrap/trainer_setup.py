from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.config import SaveDelayConfig
from src.training_runtime import resolve_training_runtime_profile
from src.trainers.metrics.mixins import (
    AggregateTokenTypeMetricsMixin,
    BBoxGeoLossMixin,
    BBoxSizeAuxLossMixin,
    CoordSoftCEW1LossMixin,
    GradAccumLossScaleMixin,
    InstabilityMonitorMixin,
    SFTStructuralCloseLossMixin,
)


def compose_trainer_class(
    *,
    trainer_cls: type,
    trainer_variant: str,
    instability_monitor_cfg: Mapping[str, Any] | None,
    token_type_cfg: Any,
    bbox_geo_cfg: Any,
    bbox_size_aux_cfg: Any,
    coord_soft_ce_w1_cfg: Any,
    sft_structural_close_cfg: Any = None,
) -> type:
    mixins: list[type] = []
    runtime_profile = resolve_training_runtime_profile(trainer_variant)
    if runtime_profile.ordinary_stage1_mixins_allowed:
        mixins.append(GradAccumLossScaleMixin)
        if isinstance(instability_monitor_cfg, Mapping) and bool(
            instability_monitor_cfg.get("enabled", False)
        ):
            mixins.append(InstabilityMonitorMixin)
        if token_type_cfg and getattr(token_type_cfg, "enabled", False):
            mixins.append(AggregateTokenTypeMetricsMixin)
        if bbox_size_aux_cfg and getattr(bbox_size_aux_cfg, "enabled", False):
            mixins.append(BBoxSizeAuxLossMixin)
        if bbox_geo_cfg and getattr(bbox_geo_cfg, "enabled", False):
            mixins.append(BBoxGeoLossMixin)
        if coord_soft_ce_w1_cfg and getattr(coord_soft_ce_w1_cfg, "enabled", False):
            mixins.append(CoordSoftCEW1LossMixin)
        if sft_structural_close_cfg and getattr(
            sft_structural_close_cfg, "enabled", False
        ):
            mixins.append(SFTStructuralCloseLossMixin)
    if not mixins:
        return trainer_cls
    return type(
        f"{trainer_cls.__name__}WithMetrics",
        tuple(mixins + [trainer_cls]),
        {},
    )


def build_trainer_callbacks(
    *,
    base_callbacks: Sequence[Any] | None,
    dataset: Any,
    append_dataset_epoch_callback_fn: Any,
    stage1_eval_detection_callback: Any,
    heartbeat_callback: Any,
    curriculum_scheduler: Any,
    curriculum_state: Any,
    save_delay_cfg: Any,
    save_delay_steps: Any,
    save_delay_epochs: Any,
    logger: Any,
) -> list[Any]:
    callbacks = list(base_callbacks or [])
    callbacks = append_dataset_epoch_callback_fn(callbacks, dataset)
    if stage1_eval_detection_callback is not None:
        callbacks.append(stage1_eval_detection_callback)
    if heartbeat_callback is not None:
        callbacks.append(heartbeat_callback)
    if curriculum_scheduler is not None and curriculum_state is not None:
        from src.callbacks.augmentation_curriculum import (
            AugmentationCurriculumCallback,
        )

        callbacks.append(
            AugmentationCurriculumCallback(
                scheduler=curriculum_scheduler,
                curriculum_state=curriculum_state,
            )
        )

    from src.callbacks import SaveDelayCallback

    if isinstance(save_delay_cfg, SaveDelayConfig) and save_delay_cfg.active:
        callbacks.append(SaveDelayCallback(config=save_delay_cfg))
        delay_info = (
            f"step {save_delay_cfg.steps}"
            if save_delay_cfg.steps is not None
            else f"epoch {save_delay_cfg.epochs}"
        )
        logger.info(
            f"SaveDelayCallback enabled: checkpoint saves blocked until {delay_info}"
        )
    else:
        if save_delay_steps is not None and save_delay_steps > 0:
            callbacks.append(SaveDelayCallback(save_delay_steps=save_delay_steps))
            logger.info(
                f"SaveDelayCallback enabled: no checkpoints until step {save_delay_steps}"
            )
        elif save_delay_epochs is not None and float(save_delay_epochs) > 0:
            callbacks.append(
                SaveDelayCallback(save_delay_epochs=float(save_delay_epochs))
            )
            logger.info(
                f"SaveDelayCallback enabled: no checkpoints until epoch {float(save_delay_epochs):.2f}"
            )

    return callbacks


def instantiate_trainer(
    *,
    trainer_cls: type,
    sft_model: Any,
    training_args: Any,
    data_collator: Any,
    dataset: Any,
    eval_dataset: Any,
    callbacks: Sequence[Any],
    template: Any,
    trainer_kwargs: Mapping[str, Any] | None,
    heartbeat_writer: Any,
) -> Any:
    if heartbeat_writer is not None:
        heartbeat_writer.emit("trainer_init_start")
    trainer = trainer_cls(
        model=sft_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=list(callbacks),
        template=template,
        **dict(trainer_kwargs or {}),
    )
    if heartbeat_writer is not None:
        heartbeat_writer.emit("trainer_init_done")
    return trainer
