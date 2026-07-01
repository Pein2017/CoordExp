"""Small token-branch trie helpers for sampled-path valid-set targets."""

from __future__ import annotations

from dataclasses import dataclass

from src.training.teacher_forcing.roles import TokenRole


@dataclass(frozen=True)
class TokenBranch:
    object_index: int
    object_instance_id: str
    token_ids: tuple[int, ...]
    token_roles: tuple[TokenRole, ...]
    coord_roles: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.token_ids)
            == len(self.token_roles)
            == len(self.coord_roles)
        ):
            raise ValueError("TokenBranch token metadata lengths must match")

    def token_id_at(self, position: int) -> int:
        return self.token_ids[position]

    def token_role_at(self, position: int) -> TokenRole:
        return self.token_roles[position]

    def coord_role_at(self, position: int) -> str | None:
        return self.coord_roles[position]


def next_token_ids_for_prefix(
    branches: tuple[TokenBranch, ...],
    *,
    position: int,
) -> frozenset[int]:
    return frozenset(branch.token_id_at(position) for branch in branches)


def token_roles_for_prefix(
    branches: tuple[TokenBranch, ...],
    *,
    position: int,
) -> frozenset[TokenRole]:
    return frozenset(branch.token_role_at(position) for branch in branches)


def filter_branches_by_selected_token(
    branches: tuple[TokenBranch, ...],
    *,
    position: int,
    selected_token_id: int,
) -> tuple[TokenBranch, ...]:
    return tuple(
        branch
        for branch in branches
        if branch.token_id_at(position) == selected_token_id
    )


__all__ = [
    "TokenBranch",
    "filter_branches_by_selected_token",
    "next_token_ids_for_prefix",
    "token_roles_for_prefix",
]
