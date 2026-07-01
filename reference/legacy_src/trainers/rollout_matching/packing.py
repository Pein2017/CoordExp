"""Rollout-matching dataloader helpers.

Stage-2 standardized on micro-scope dynamic packing. The legacy window-aware
packing/lookahead helpers have been removed to keep the training surface small.

The remaining wrapper is used to keep optimizer-step semantics consistent when
`training.dataloader_drop_last=true` and `gradient_accumulation_steps>1`.
"""

from __future__ import annotations


class DropRemainderAccumulationWindow:
    """Drop the final partial gradient-accumulation window.

    HF/Swift will still perform an optimizer step on a partial accumulation window at
    the end of an epoch. For stage-2 step-budgeted trainers we typically want fixed
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
    "DropRemainderAccumulationWindow",
]
