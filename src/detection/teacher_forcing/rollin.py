"""Deterministic roll-in order for compact_full teacher-forcing targets."""

from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

ROLLIN_POLICY_NAME = "random_permutation"
ROLLIN_POLICY_VERSION = 1
DEFAULT_ROLLIN_BASE_SEED = 17

T = TypeVar("T")


def derive_rollin_seed(
    *,
    base_seed: int = DEFAULT_ROLLIN_BASE_SEED,
    epoch: int,
    stable_sample_id: str,
    policy_name: str = ROLLIN_POLICY_NAME,
    policy_version: int = ROLLIN_POLICY_VERSION,
) -> int:
    payload = "|".join(
        (
            str(int(base_seed)),
            str(int(epoch)),
            str(stable_sample_id),
            str(policy_name),
            str(int(policy_version)),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def random_permutation_rollin(
    items: Sequence[T],
    *,
    seed: int,
) -> tuple[T, ...]:
    ordered = list(items)
    random.Random(int(seed)).shuffle(ordered)
    return tuple(ordered)


def ordered_rollin(items: Sequence[T]) -> tuple[T, ...]:
    return tuple(items)


__all__ = [
    "DEFAULT_ROLLIN_BASE_SEED",
    "ROLLIN_POLICY_NAME",
    "ROLLIN_POLICY_VERSION",
    "derive_rollin_seed",
    "ordered_rollin",
    "random_permutation_rollin",
]
