"""Token role resolution for trainable rows and loss masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class TokenRole(str, Enum):
    COORD_GEOMETRY = "coord_geometry"
    STRUCTURAL_CE_ONLY = "structural_ce_only"


def normalize_token_role(value: object, *, path: str) -> TokenRole:
    if isinstance(value, TokenRole):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    try:
        return TokenRole(value)
    except ValueError as exc:
        allowed = ", ".join(role.value for role in TokenRole)
        raise ValueError(f"{path} must be one of [{allowed}], got {value!r}") from exc


def unique_stable_ids(values: Iterable[int]) -> tuple[int, ...]:
    resolved: list[int] = []
    seen: set[int] = set()
    for value in values:
        token_id = int(value)
        if token_id in seen:
            continue
        resolved.append(token_id)
        seen.add(token_id)
    return tuple(resolved)


@dataclass(frozen=True)
class TokenRoleSets:
    coord_geometry_ids: tuple[int, ...] = ()
    structural_ce_only_ids: tuple[int, ...] = ()
    trainable_row_ids: tuple[int, ...] = ()
    coord_loss_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        coord_geometry_ids = unique_stable_ids(self.coord_geometry_ids)
        structural_ce_only_ids = unique_stable_ids(self.structural_ce_only_ids)
        if set(coord_geometry_ids) & set(structural_ce_only_ids):
            overlap = sorted(set(coord_geometry_ids) & set(structural_ce_only_ids))
            raise ValueError(
                "Token role sets must keep coord_geometry and structural_ce_only "
                f"disjoint; overlap={overlap[:8]}"
            )

        trainable_row_ids = (
            unique_stable_ids(self.trainable_row_ids)
            if self.trainable_row_ids
            else unique_stable_ids((*coord_geometry_ids, *structural_ce_only_ids))
        )
        coord_loss_ids = (
            unique_stable_ids(self.coord_loss_ids)
            if self.coord_loss_ids
            else coord_geometry_ids
        )
        if set(coord_loss_ids) - set(coord_geometry_ids):
            bad = sorted(set(coord_loss_ids) - set(coord_geometry_ids))
            raise ValueError(
                "Coordinate loss ids must be a subset of coord_geometry ids; "
                f"unexpected={bad[:8]}"
            )
        if set(coord_loss_ids) & set(structural_ce_only_ids):
            bad = sorted(set(coord_loss_ids) & set(structural_ce_only_ids))
            raise ValueError(
                "structural_ce_only ids must not enter coordinate loss ids; "
                f"overlap={bad[:8]}"
            )

        object.__setattr__(self, "coord_geometry_ids", coord_geometry_ids)
        object.__setattr__(self, "structural_ce_only_ids", structural_ce_only_ids)
        object.__setattr__(self, "trainable_row_ids", trainable_row_ids)
        object.__setattr__(self, "coord_loss_ids", coord_loss_ids)


__all__ = [
    "TokenRole",
    "TokenRoleSets",
    "normalize_token_role",
    "unique_stable_ids",
]
