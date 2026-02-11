"""Rollout-matching packing/windowing helpers.

These utilities provide a stable contract for window-aware gradient accumulation
("post-rollout packing"), without requiring imports from trainer implementation
modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RolloutMatchingPackWindow:
    """Holds one accumulation window worth of raw micro-batches and cached prepared batches.

    This enables window-aware scheduling (lookahead) without changing the Trainer's
    micro-step interface: each yielded micro-batch is still a list, but carries a
    pointer to the full window.
    """

    def __init__(self, *, raw_micro_batches: List[List[Any]]):
        self.raw_micro_batches = raw_micro_batches
        self._prepared_micro_batches: Optional[List[Dict[str, Any]]] = None

    @property
    def gas(self) -> int:
        return int(len(self.raw_micro_batches))

    def get_prepared(
        self,
        *,
        idx: int,
        build_all_prepared: Any,
    ) -> Dict[str, Any]:
        if self._prepared_micro_batches is None:
            self._prepared_micro_batches = list(build_all_prepared())
        if not isinstance(self._prepared_micro_batches, list):
            raise ValueError("prepared window batches must be a list")
        if idx < 0 or idx >= len(self._prepared_micro_batches):
            raise IndexError("prepared window batch index out of range")
        prepared = self._prepared_micro_batches[idx]
        if not isinstance(prepared, dict):
            raise ValueError("prepared batch must be a dict")
        return prepared


class WindowedMicroBatch(list):
    """List-like micro-batch carrying a reference to its full accumulation window."""

    def __init__(
        self,
        raw: List[Any],
        *,
        window: RolloutMatchingPackWindow,
        idx: int,
    ) -> None:
        super().__init__(raw)
        self.rm_window = window
        self.rm_window_idx = int(idx)


class AccumulationWindowLookahead:
    """Prefetch `gas` micro-batches so the trainer can schedule within the full window."""

    def __init__(self, dataloader, *, gas: int):
        self.dataloader = dataloader
        self.gas = int(gas)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        it = iter(self.dataloader)
        while True:
            raw_window: List[List[Any]] = []
            try:
                for _ in range(int(self.gas)):
                    b = next(it)
                    # Identity collator yields a list of raw samples.
                    if not isinstance(b, list):
                        raise ValueError(
                            "window lookahead expects identity-collated train batches (list of raw samples)"
                        )
                    raw_window.append(b)
            except StopIteration:
                # Partial final window: yield whatever is left without window context.
                for b in raw_window:
                    yield b
                break

            window = RolloutMatchingPackWindow(raw_micro_batches=raw_window)
            for i, b in enumerate(raw_window):
                yield WindowedMicroBatch(b, window=window, idx=int(i))


class DropRemainderAccumulationWindow:
    """Drop the final partial gradient-accumulation window.

    HF/Swift will still perform an optimizer step on a partial accumulation window at
    the end of an epoch. For stage_2 step-budgeted trainers we typically want fixed
    raw-sample budgets per optimizer step, so when `training.dataloader_drop_last` is
    enabled we truncate the train dataloader to a multiple of
    `gradient_accumulation_steps`.

    This is intentionally a lightweight wrapper that preserves epoch semantics by
    dropping the remainder *per epoch*.
    """

    def __init__(self, dataloader, *, gas: int):
        self.dataloader = dataloader
        self.gas = max(1, int(gas))

    def __len__(self) -> int:
        inner_len = getattr(self.dataloader, "__len__", None)
        if inner_len is None:
            raise TypeError("wrapped dataloader has no __len__")
        n = int(len(self.dataloader))
        return int((n // int(self.gas)) * int(self.gas))

    def __iter__(self):
        n_keep = int(len(self))
        for i, b in enumerate(self.dataloader):
            if i >= n_keep:
                break
            yield b

    def __getattr__(self, name: str):
        return getattr(self.dataloader, name)


__all__ = [
    "RolloutMatchingPackWindow",
    "WindowedMicroBatch",
    "AccumulationWindowLookahead",
    "DropRemainderAccumulationWindow",
]
