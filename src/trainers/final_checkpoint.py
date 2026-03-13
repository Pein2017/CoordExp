"""Trainer checkpoint-state adapters for upstream save behavior gaps."""

from __future__ import annotations

import logging
import os
import weakref
from typing import TYPE_CHECKING, Dict, Type, cast

from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy

if TYPE_CHECKING:  # pragma: no cover
    from transformers import Trainer as _TrainerBase


logger = logging.getLogger(__name__)


class _FinalCheckpointCallback(TrainerCallback):
    """Callback bound to a specific trainer instance to enforce the final save."""

    def __init__(self, owner: "FinalCheckpointMixin") -> None:
        self._owner_ref = weakref.ref(owner)

    def on_train_end(  # type: ignore[override]
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        owner = self._owner_ref()
        if owner is None:
            return control
        owner._maybe_save_final_checkpoint(args, state, control)
        return control


class FinalCheckpointMixin:
    """Adds checkpoint-state repairs and optional final-save fallback."""

    _final_checkpoint_callback_attr = "_final_checkpoint_callback"
    _final_checkpoint_wrapper_cache: Dict[Type, Type] = {}
    _pending_best_checkpoint_step_attr = "_pending_best_checkpoint_step"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        setattr(self, self._pending_best_checkpoint_step_attr, None)
        if (
            not hasattr(self, self._final_checkpoint_callback_attr)
            and self._should_install_final_checkpoint_callback()
        ):
            callback = _FinalCheckpointCallback(self)
            setattr(self, self._final_checkpoint_callback_attr, callback)
            trainer = cast("_TrainerBase", self)
            trainer.add_callback(callback)

    def _determine_best_metric(self, metrics, trial):
        """Track best-checkpoint saves for `save_strategy='best'`.

        Upstream `transformers` saves the checkpoint when a new best metric is found,
        but does not populate `state.best_model_checkpoint` for `SaveStrategy.BEST`.
        Record the step here so `_save_checkpoint()` can stamp the correct path after
        the checkpoint directory exists on disk.
        """

        setattr(self, self._pending_best_checkpoint_step_attr, None)
        is_new_best_metric = super()._determine_best_metric(metrics, trial)

        if getattr(self.args, "save_strategy", SaveStrategy.NO) != SaveStrategy.BEST:
            return is_new_best_metric
        if not is_new_best_metric:
            return is_new_best_metric

        global_step = getattr(self.state, "global_step", None)
        if isinstance(global_step, int) and global_step > 0:
            setattr(self, self._pending_best_checkpoint_step_attr, global_step)
        return is_new_best_metric

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        """Preserve best-checkpoint pointers for best-only save cadence."""

        super()._save_checkpoint(model, trial, *args, **kwargs)

        checkpoint_dir = self._get_current_checkpoint_dir(trial=trial)
        if checkpoint_dir is None:
            setattr(self, self._pending_best_checkpoint_step_attr, None)
            return

        if self._record_best_checkpoint_if_pending(checkpoint_dir):
            if getattr(self.args, "should_save", False):
                self.state.save_to_json(os.path.join(checkpoint_dir, TRAINER_STATE_NAME))

    # ------------------------------------------------------------------
    # Final checkpoint helpers
    # ------------------------------------------------------------------
    def _should_install_final_checkpoint_callback(self) -> bool:
        """Return True when the repo-owned final-save fallback should be active."""

        trainer = cast("_TrainerBase", self)
        save_last_epoch = bool(getattr(trainer.args, "save_last_epoch", True))
        if not save_last_epoch:
            return False
        return self._save_strategy_requires_final_checkpoint_fallback(
            getattr(trainer.args, "save_strategy", SaveStrategy.NO)
        )

    @staticmethod
    def _save_strategy_requires_final_checkpoint_fallback(save_strategy) -> bool:
        """Return True when upstream may skip the terminal checkpoint save."""

        return save_strategy == SaveStrategy.BEST

    def _maybe_save_final_checkpoint(  # noqa: PLR0912
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
    ) -> None:
        """Persist the last checkpoint if the training loop skipped it."""

        trainer = cast("_TrainerBase", self)

        save_strategy = getattr(args, "save_strategy", SaveStrategy.NO)
        if not self._save_strategy_requires_final_checkpoint_fallback(save_strategy):
            logger.debug(
                "Final checkpoint fallback skipped because save_strategy=%s is handled upstream.",
                save_strategy,
            )
            return

        should_save_rank = bool(getattr(args, "should_save", False))
        world_size = getattr(args, "world_size", 1)

        if not should_save_rank:
            if world_size <= 1:
                logger.debug(
                    "Final checkpoint skipped because no process is permitted to save checkpoints."
                )
                return
            logger.debug(
                "Final checkpoint: this rank will participate in the distributed save without writing to disk."
            )

        global_step = getattr(state, "global_step", 0)
        if not isinstance(global_step, int) or global_step <= 0:
            logger.debug("Final checkpoint skipped because global_step=%s", global_step)
            return

        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            logger.debug("Final checkpoint skipped because output_dir is undefined.")
            return

        if self._final_checkpoint_exists(output_dir, global_step):
            if should_save_rank:
                logger.debug("Final checkpoint already present for step %s", global_step)
            return

        checkpoint_dir = self._format_checkpoint_dir(output_dir, global_step)
        if should_save_rank:
            logger.info("No checkpoint found at %s; forcing a final save.", checkpoint_dir)

        # Keep the forced checkpoint independent from Trainer-managed rotation so
        # save_total_limit continues to govern only the regular save cadence.
        original_limit = getattr(args, "save_total_limit", None)
        limit_suspended = (
            should_save_rank
            and isinstance(original_limit, int)
            and original_limit > 0
        )
        if limit_suspended:
            setattr(args, "save_total_limit", None)
            logger.info(
                "Temporarily disabling save_total_limit=%s while writing the final checkpoint.",
                original_limit,
            )

        try:
            try:
                trainer._save_checkpoint(trainer.model, None)  # type: ignore[misc,arg-type]
            except TypeError:
                # Some trainer overrides accept metrics; fall back to keyword form.
                trainer._save_checkpoint(trainer.model, None, metrics=None)  # type: ignore[misc,call-arg]
        finally:
            if limit_suspended:
                setattr(args, "save_total_limit", original_limit)

        # Mirror Trainer.train() behaviour so callbacks observe the save event.
        trainer.callback_handler.on_save(args, state, control)

    def _record_best_checkpoint_if_pending(self, checkpoint_dir: str) -> bool:
        """Update best-checkpoint state when the just-saved checkpoint is the new best."""

        pending_step = getattr(self, self._pending_best_checkpoint_step_attr, None)
        global_step = getattr(self.state, "global_step", None)
        setattr(self, self._pending_best_checkpoint_step_attr, None)

        if getattr(self.args, "save_strategy", SaveStrategy.NO) != SaveStrategy.BEST:
            return False
        if not isinstance(global_step, int) or global_step <= 0:
            return False
        if pending_step != global_step:
            return False
        if not os.path.isdir(checkpoint_dir):
            return False

        self.state.best_model_checkpoint = checkpoint_dir
        self.state.best_global_step = global_step
        return True

    def _get_current_checkpoint_dir(self, trial) -> str | None:
        """Return the on-disk path for the current checkpoint save."""

        trainer = cast("_TrainerBase", self)
        global_step = getattr(self.state, "global_step", None)
        if not isinstance(global_step, int) or global_step <= 0:
            return None

        run_dir = trainer._get_output_dir(trial=trial)
        return self._format_checkpoint_dir(run_dir, global_step)

    def _final_checkpoint_exists(self, output_dir: str, step: int) -> bool:
        """Return True if the checkpoint directory (or flash record) already exists."""

        trainer = cast("_TrainerBase", self)

        if getattr(trainer.args, "use_flash_ckpt", False) and hasattr(
            trainer, "_get_last_checkpoint_step"
        ):
            last_step = trainer._get_last_checkpoint_step()  # type: ignore[attr-defined]
            if isinstance(last_step, int) and last_step >= step:
                return True

        checkpoint_dir = self._format_checkpoint_dir(output_dir, step)
        if os.path.isdir(checkpoint_dir):
            return True

        last_model_checkpoint = getattr(trainer.state, "last_model_checkpoint", None)
        if isinstance(last_model_checkpoint, str) and os.path.isdir(
            last_model_checkpoint
        ):
            if os.path.basename(last_model_checkpoint) == os.path.basename(
                checkpoint_dir
            ):
                return True

        return False

    @staticmethod
    def _format_checkpoint_dir(output_dir: str, step: int) -> str:
        return os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")


def with_final_checkpoint(trainer_cls: Type) -> Type:
    """Return a trainer subclass that includes :class:`FinalCheckpointMixin`."""

    if not isinstance(trainer_cls, type):
        raise TypeError("trainer_cls must be a class")

    if issubclass(trainer_cls, FinalCheckpointMixin):
        return trainer_cls

    cache = FinalCheckpointMixin._final_checkpoint_wrapper_cache
    if trainer_cls in cache:
        return cache[trainer_cls]

    wrapped = type(
        f"{trainer_cls.__name__}WithFinalCheckpoint",
        (FinalCheckpointMixin, trainer_cls),
        {},
    )
    wrapped.__module__ = trainer_cls.__module__
    cache[trainer_cls] = wrapped
    return wrapped


__all__ = ["FinalCheckpointMixin", "with_final_checkpoint"]
