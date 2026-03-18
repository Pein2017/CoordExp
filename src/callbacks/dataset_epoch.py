from __future__ import annotations

from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class DatasetEpochCallback(TrainerCallback):
    """Advance datasets that expose set_epoch(...) at each trainer epoch boundary."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self._last_epoch: int | None = None

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        epoch = (
            int(state.epoch)
            if state.epoch is not None
            else int(state.global_step or 0)
        )
        if self._last_epoch == epoch:
            return
        set_epoch_fn = getattr(self.dataset, "set_epoch", None)
        if callable(set_epoch_fn):
            set_epoch_fn(epoch)
        self._last_epoch = epoch
