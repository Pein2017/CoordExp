from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .roles import TokenRole


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


@dataclass(frozen=True)
class TeacherForcingTargetIR:
    schema_version: int
    atoms: tuple[SupervisionAtom, ...]
    metadata: Mapping[str, Any]
