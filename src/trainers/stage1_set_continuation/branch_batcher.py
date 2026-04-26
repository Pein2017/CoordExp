from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence, cast


@dataclass(frozen=True)
class BranchBatchWorkItem:
    index: int
    sequence_length: int
    suffix_keep: int

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("branch batch work item index must be >= 0")
        if int(self.sequence_length) <= 0:
            raise ValueError("branch batch work item sequence_length must be > 0")
        if int(self.suffix_keep) <= 0:
            raise ValueError("branch batch work item suffix_keep must be > 0")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "sequence_length", int(self.sequence_length))
        object.__setattr__(self, "suffix_keep", int(self.suffix_keep))


@dataclass(frozen=True)
class BranchBatch:
    items: tuple[BranchBatchWorkItem, ...]
    real_token_volume: int
    padded_token_volume: int

    @classmethod
    def from_items(cls, items: Sequence[BranchBatchWorkItem]) -> "BranchBatch":
        resolved = tuple(items)
        if not resolved:
            raise ValueError("branch batch requires at least one work item")
        real = sum(item.sequence_length for item in resolved)
        padded = max(item.sequence_length for item in resolved) * len(resolved)
        return cls(
            items=resolved,
            real_token_volume=int(real),
            padded_token_volume=int(padded),
        )


@dataclass(frozen=True)
class SmartBranchBatchPlan:
    batches: tuple[BranchBatch, ...]
    scheduler: str
    total_real_tokens: int
    total_padded_tokens: int

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    @property
    def padding_fraction(self) -> float:
        if self.total_padded_tokens <= 0:
            return 0.0
        waste = max(0, self.total_padded_tokens - self.total_real_tokens)
        return float(waste / self.total_padded_tokens)

    @property
    def rows_mean(self) -> float:
        if not self.batches:
            return 0.0
        return float(
            sum(len(batch.items) for batch in self.batches) / len(self.batches)
        )

    @property
    def rows_max(self) -> float:
        if not self.batches:
            return 0.0
        return float(max(len(batch.items) for batch in self.batches))

    @property
    def tokens_mean(self) -> float:
        if not self.batches:
            return 0.0
        return float(
            sum(batch.padded_token_volume for batch in self.batches) / len(self.batches)
        )

    @property
    def tokens_max(self) -> float:
        if not self.batches:
            return 0.0
        return float(max(batch.padded_token_volume for batch in self.batches))


def _positive_or_none(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0 when provided")
    return parsed


def _fits(
    items: Sequence[BranchBatchWorkItem],
    *,
    max_branch_rows: int | None,
    max_branch_tokens: int | None,
) -> bool:
    if not items:
        return True
    if max_branch_rows is not None and len(items) > max_branch_rows:
        return False
    if max_branch_tokens is not None:
        padded = max(item.sequence_length for item in items) * len(items)
        if padded > max_branch_tokens:
            return False
    return True


def _split_group(
    items: Sequence[BranchBatchWorkItem],
    *,
    max_branch_rows: int | None,
    max_branch_tokens: int | None,
) -> list[BranchBatch]:
    batches: list[BranchBatch] = []
    current: list[BranchBatchWorkItem] = []
    for item in sorted(items, key=lambda entry: (-entry.sequence_length, entry.index)):
        proposed = [*current, item]
        if current and not _fits(
            proposed,
            max_branch_rows=max_branch_rows,
            max_branch_tokens=max_branch_tokens,
        ):
            batches.append(BranchBatch.from_items(current))
            current = [item]
        else:
            current = proposed
    if current:
        batches.append(BranchBatch.from_items(current))
    return batches


def _constant_volume_groups(
    items: Sequence[BranchBatchWorkItem],
    *,
    max_branch_tokens: int | None,
) -> tuple[list[list[BranchBatchWorkItem]], str]:
    if max_branch_tokens is None:
        return [list(items)], "deterministic_fallback"
    try:
        import binpacking  # type: ignore[import-untyped]

        work = [(pos, item.sequence_length) for pos, item in enumerate(items)]
        to_constant_volume = cast(
            Callable[..., Any],
            getattr(binpacking, "to_constant_volume"),
        )
        packed = cast(
            list[list[tuple[int, int]]],
            to_constant_volume(work, int(max_branch_tokens), weight_pos=1),
        )
        groups: list[list[BranchBatchWorkItem]] = []
        for group in packed:
            groups.append([items[int(pos)] for pos, _length in group])
        return groups, "constant_volume"
    except Exception:
        return [list(items)], "deterministic_fallback"


def plan_smart_branch_batches(
    items: Sequence[BranchBatchWorkItem],
    *,
    max_branch_rows: int | None = None,
    max_branch_tokens: int | None = None,
    min_fill_ratio: float = 0.70,
) -> SmartBranchBatchPlan:
    del min_fill_ratio  # Reserved for the later adaptive scheduler.
    resolved = tuple(items)
    if not resolved:
        return SmartBranchBatchPlan(
            batches=(),
            scheduler="disabled",
            total_real_tokens=0,
            total_padded_tokens=0,
        )
    row_cap = _positive_or_none(max_branch_rows, name="max_branch_rows")
    token_cap = _positive_or_none(max_branch_tokens, name="max_branch_tokens")
    groups, scheduler = _constant_volume_groups(resolved, max_branch_tokens=token_cap)
    batches: list[BranchBatch] = []
    for group in groups:
        batches.extend(
            _split_group(
                group,
                max_branch_rows=row_cap,
                max_branch_tokens=token_cap,
            )
        )
    batches = sorted(
        batches,
        key=lambda batch: min(item.index for item in batch.items),
    )
    total_real = sum(batch.real_token_volume for batch in batches)
    total_padded = sum(batch.padded_token_volume for batch in batches)
    return SmartBranchBatchPlan(
        batches=tuple(batches),
        scheduler=scheduler,
        total_real_tokens=int(total_real),
        total_padded_tokens=int(total_padded),
    )


__all__ = [
    "BranchBatch",
    "BranchBatchWorkItem",
    "SmartBranchBatchPlan",
    "plan_smart_branch_batches",
]
