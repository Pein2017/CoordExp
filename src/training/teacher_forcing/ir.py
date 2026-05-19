from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .roles import TokenRole


def _immutable_mapping(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class SupervisionAtom:
    batch_index: int
    logit_position: int
    target_position: int
    allowed_token_roles: frozenset[TokenRole]
    selected_token_role: TokenRole
    valid_token_ids: frozenset[int]
    selected_token_id: int
    latent_valid_token_ids: frozenset[int]
    coverage_target_weights: Mapping[int, float] | None
    loss_tags: frozenset[str]
    loss_weight: float
    coord_role: str | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_token_roles", frozenset(self.allowed_token_roles))
        object.__setattr__(self, "valid_token_ids", frozenset(self.valid_token_ids))
        object.__setattr__(self, "latent_valid_token_ids", frozenset(self.latent_valid_token_ids))
        if self.coverage_target_weights is not None:
            object.__setattr__(
                self,
                "coverage_target_weights",
                _immutable_mapping(self.coverage_target_weights),
            )
        object.__setattr__(self, "loss_tags", frozenset(self.loss_tags))
        object.__setattr__(self, "provenance", _immutable_mapping(self.provenance))


@dataclass(frozen=True)
class TeacherForcingTargetIR:
    schema_version: int
    atoms: tuple[SupervisionAtom, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
