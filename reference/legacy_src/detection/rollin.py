from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NewType, Sequence


ObjectInstanceId = NewType("ObjectInstanceId", str)


@dataclass(frozen=True)
class RollinState:
    k: int
    permutation: tuple[ObjectInstanceId, ...]
    emitted: tuple[ObjectInstanceId, ...]
    remaining: tuple[ObjectInstanceId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "permutation", tuple(self.permutation))
        object.__setattr__(self, "emitted", tuple(self.emitted))
        object.__setattr__(self, "remaining", tuple(self.remaining))
        if type(self.k) is not int:
            raise TypeError("k must be an integer")
        if len(set(self.permutation)) != len(self.permutation):
            raise ValueError("permutation contains duplicate object_instance_ids")
        if self.k < 0 or self.k > len(self.permutation):
            raise ValueError(f"k must satisfy 0 <= k <= {len(self.permutation)}")
        if self.emitted != self.permutation[: self.k]:
            raise ValueError("emitted must equal permutation[:k]")
        if self.remaining != self.permutation[self.k :]:
            raise ValueError("remaining must equal permutation[k:]")


def _validate_unique_object_instance_ids(
    object_instance_ids: Sequence[ObjectInstanceId],
) -> tuple[ObjectInstanceId, ...]:
    objects = tuple(object_instance_ids)
    if len(set(objects)) != len(objects):
        raise ValueError("object_instance_ids contains duplicate object_instance_ids")
    return objects


def make_prefix_rollin_state(
    object_instance_ids: Sequence[ObjectInstanceId],
    *,
    permutation: Sequence[ObjectInstanceId],
    k: int,
) -> RollinState:
    objects = _validate_unique_object_instance_ids(object_instance_ids)

    realized_permutation = tuple(permutation)
    if len(realized_permutation) != len(objects) or set(realized_permutation) != set(
        objects
    ):
        raise ValueError("permutation must contain each object_instance_id exactly once")

    if len(set(realized_permutation)) != len(realized_permutation):
        raise ValueError("permutation must contain each object_instance_id exactly once")

    if type(k) is not int:
        raise TypeError("k must be an integer")
    if k < 0 or k > len(objects):
        raise ValueError(f"k must satisfy 0 <= k <= {len(objects)}")

    emitted = realized_permutation[:k]
    remaining = realized_permutation[k:]
    return RollinState(
        k=k,
        permutation=realized_permutation,
        emitted=emitted,
        remaining=remaining,
    )


def sample_prefix_rollin_state(
    object_instance_ids: Sequence[ObjectInstanceId],
    *,
    rng: random.Random,
) -> RollinState:
    objects = _validate_unique_object_instance_ids(object_instance_ids)
    permutation = tuple(rng.sample(objects, k=len(objects)))
    k = rng.randint(0, len(permutation))
    return make_prefix_rollin_state(objects, permutation=permutation, k=k)
