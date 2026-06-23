"""Shared row-selection helpers for detection debug and preflight datasets."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SEEDED_RANDOM_WITHOUT_REPLACEMENT = "seeded_random_without_replacement_v0"


@dataclass(frozen=True)
class DatasetRowSelectionConfig:
    algorithm: str
    count: int
    seed: int


def normalize_dataset_row_selection(
    value: Any,
    *,
    path: str,
) -> DatasetRowSelectionConfig | None:
    if value is None:
        return None
    if isinstance(value, DatasetRowSelectionConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping when provided")
    data = dict(value)
    algorithm = str(data.pop("algorithm", "") or "")
    count = data.pop("count", None)
    seed = data.pop("seed", None)
    if data:
        unknown = ", ".join(sorted(str(key) for key in data))
        raise ValueError(f"Unknown {path} keys: {unknown}")
    if algorithm != SEEDED_RANDOM_WITHOUT_REPLACEMENT:
        raise ValueError(
            f"{path}.algorithm must be {SEEDED_RANDOM_WITHOUT_REPLACEMENT!r}"
        )
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"{path}.count must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path}.seed must be an integer")
    return DatasetRowSelectionConfig(
        algorithm=algorithm,
        count=int(count),
        seed=int(seed),
    )


def select_dataset_row_indices(
    *,
    total_rows: int,
    selection: DatasetRowSelectionConfig | Mapping[str, Any],
) -> tuple[int, ...]:
    config = normalize_dataset_row_selection(selection, path="selection")
    if config is None:
        raise ValueError("selection is required")
    if total_rows < config.count:
        raise ValueError(
            f"row selection requires {config.count} rows but source JSONL has {total_rows}"
        )
    if config.algorithm == SEEDED_RANDOM_WITHOUT_REPLACEMENT:
        rng = random.Random(config.seed)
        return tuple(rng.sample(range(int(total_rows)), k=config.count))
    raise ValueError(f"Unsupported row selection algorithm: {config.algorithm!r}")


__all__ = [
    "DatasetRowSelectionConfig",
    "SEEDED_RANDOM_WITHOUT_REPLACEMENT",
    "normalize_dataset_row_selection",
    "select_dataset_row_indices",
]
