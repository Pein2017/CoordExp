from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.config import SaveDelayConfig
from src.training_runtime import resolve_training_runtime_profile
from src.trainers.metrics.mixins import (
    AggregateTokenTypeMetricsMixin,
    CoordSoftCEW1LossMixin,
    GradAccumLossScaleMixin,
    InstabilityMonitorMixin,
    PrefixDenoisingObjectiveMixin,
    RecursiveDetectionCEMixin,
    SFTStructuralCloseLossMixin,
    TeacherForcingObjectiveMixin,
)


class _InjectedSwiftDataCollatorMixin:
    """Route CoordExp's collator through ms-swift's current trainer lifecycle."""

    def __init__(self, *args: Any, data_collator: Any = None, **kwargs: Any) -> None:
        self._coordexp_injected_data_collator = data_collator
        super().__init__(*args, **kwargs)  # type: ignore[misc]

    def _get_data_collator(self, args: Any, template: Any) -> Any:
        collator = getattr(self, "_coordexp_injected_data_collator", None)
        if collator is not None:
            return collator
        return super()._get_data_collator(args, template)  # type: ignore[misc]


_SWIFT_COLLATOR_WRAPPER_CACHE: dict[type, type] = {}


def _trainer_uses_swift_collator_factory(trainer_cls: type) -> bool:
    for cls in trainer_cls.mro():
        if str(getattr(cls, "__module__", "")).startswith("swift.") and hasattr(
            cls, "_get_data_collator"
        ):
            return True
    return False


def _with_injected_swift_data_collator(trainer_cls: type) -> type:
    if not _trainer_uses_swift_collator_factory(trainer_cls):
        return trainer_cls
    if issubclass(trainer_cls, _InjectedSwiftDataCollatorMixin):
        return trainer_cls
    cached = _SWIFT_COLLATOR_WRAPPER_CACHE.get(trainer_cls)
    if cached is not None:
        return cached
    wrapped = type(
        f"{trainer_cls.__name__}WithInjectedDataCollator",
        (_InjectedSwiftDataCollatorMixin, trainer_cls),
        {},
    )
    wrapped.__module__ = trainer_cls.__module__
    _SWIFT_COLLATOR_WRAPPER_CACHE[trainer_cls] = wrapped
    return wrapped


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
    recursive_detection_ce_cfg: Any = None,
    teacher_forcing_objective_cfg: Any = None,
    prefix_denoising_cfg: Any = None,
    prefix_denoising_runtime: Mapping[str, Any] | None = None,
) -> type:
    mixins: list[type] = []
    class_attrs: dict[str, Any] = {}
    runtime_profile = resolve_training_runtime_profile(trainer_variant)
    if runtime_profile.ordinary_stage1_mixins_allowed:
        prefix_denoising_enabled = bool(
            prefix_denoising_cfg and getattr(prefix_denoising_cfg, "enabled", False)
        )
        recursive_ce_enabled = bool(
            recursive_detection_ce_cfg
            and getattr(recursive_detection_ce_cfg, "enabled", False)
        )
        teacher_forcing_enabled = bool(
            teacher_forcing_objective_cfg
            and getattr(teacher_forcing_objective_cfg, "enabled", True)
        )
        if prefix_denoising_enabled and recursive_ce_enabled:
            raise ValueError(
                "prefix_denoising and recursive_detection_ce are mutually exclusive "
                "token-loss owners"
            )
        if prefix_denoising_enabled:
            if not isinstance(prefix_denoising_runtime, Mapping):
                raise ValueError(
                    "prefix_denoising enabled but prefix_denoising_runtime is missing"
                )
            if "packing_enabled" not in prefix_denoising_runtime:
                raise ValueError(
                    "prefix_denoising_runtime must include packing_enabled"
                )
            class_attrs["prefix_denoising_packing_enabled"] = bool(
                prefix_denoising_runtime["packing_enabled"]
            )
        mixins.append(GradAccumLossScaleMixin)
        if prefix_denoising_enabled or recursive_ce_enabled or teacher_forcing_enabled:
            incompatible = []
            for name, cfg in (
                ("bbox_size_aux", bbox_size_aux_cfg),
                ("bbox_geo", bbox_geo_cfg),
                ("coord_soft_ce_w1", coord_soft_ce_w1_cfg),
                ("sft_structural_close", sft_structural_close_cfg),
            ):
                if cfg and getattr(cfg, "enabled", False):
                    incompatible.append(name)
            if incompatible:
                joined = ", ".join(sorted(incompatible))
                raise ValueError(
                    "teacher-forced target sidecars currently own the token loss "
                    "and do not support auxiliary loss mixins in the same "
                    f"trainer composition: {joined}"
                )
        if isinstance(instability_monitor_cfg, Mapping) and bool(
            instability_monitor_cfg.get("enabled", False)
        ):
            mixins.append(InstabilityMonitorMixin)
        if token_type_cfg and getattr(token_type_cfg, "enabled", False):
            mixins.append(AggregateTokenTypeMetricsMixin)
        if prefix_denoising_enabled:
            mixins.append(PrefixDenoisingObjectiveMixin)
        elif recursive_ce_enabled:
            mixins.append(RecursiveDetectionCEMixin)
        elif teacher_forcing_enabled:
            mixins.append(TeacherForcingObjectiveMixin)
        if (
            not prefix_denoising_enabled
            and not recursive_ce_enabled
            and coord_soft_ce_w1_cfg
            and getattr(coord_soft_ce_w1_cfg, "enabled", False)
        ):
            mixins.append(CoordSoftCEW1LossMixin)
        if (
            not prefix_denoising_enabled
            and not recursive_ce_enabled
            and sft_structural_close_cfg
            and getattr(sft_structural_close_cfg, "enabled", False)
        ):
            mixins.append(SFTStructuralCloseLossMixin)
    if not mixins:
        return trainer_cls
    return type(
        f"{trainer_cls.__name__}WithMetrics",
        tuple(mixins + [trainer_cls]),
        class_attrs,
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
    trainer_cls = _with_injected_swift_data_collator(trainer_cls)
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
