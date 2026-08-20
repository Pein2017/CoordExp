"""Private frozen inventory of the implemented supervised token losses.

This module is **metadata for the single deep loss-runner implementation**
(`src.losses.runner.LossRunner`), not a second runtime abstraction and not an
extension point. It is deliberately excluded from the public `src.losses`
package surface: there is no registry, no discovery, no import-by-name, no
callable configuration, and no pass-through factory. The repository has
exactly three implemented token losses and they are compiled with the code.

Each binding declares:

- `name`      -- the canonical term name used by configs, denominators,
                 bundles, and telemetry;
- `role`      -- `protected` (always composed) or `auxiliary` (optional);
- `normalizer`-- the planned-step reducer the term must use;
- `zero_policy`:
    * `forbid`               -- the term may never be zeroed or reweighted;
    * `detached_diagnostic`  -- at weight zero the term is still computed, but
                               entirely outside autograd, contributes a
                               literal zero weighted value, and keeps its raw
                               / count / finite diagnostics;
    * `omit`                 -- at weight zero the term is not instantiated,
                               builds no denominator, runs no math, and emits
                               no fields at all.

Adding a fourth token loss is a source change here plus an explicit branch in
the runner's closed composition path, never a configuration or plugin action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.common.errors import LossContractError

TokenLossRole = Literal["protected", "auxiliary"]
TokenLossNormalizer = Literal["segment_balanced"]
TokenLossZeroPolicy = Literal["forbid", "detached_diagnostic", "omit"]


@dataclass(frozen=True, slots=True)
class TokenLossBinding:
    """Immutable metadata for one implemented token loss."""

    name: str
    role: TokenLossRole
    normalizer: TokenLossNormalizer
    zero_policy: TokenLossZeroPolicy


BASE_CE_BINDING = TokenLossBinding(
    name="base_ce",
    role="protected",
    normalizer="segment_balanced",
    zero_policy="forbid",
)

TOKEN_TYPE_GATE_BINDING = TokenLossBinding(
    name="token_type_gate",
    role="protected",
    normalizer="segment_balanced",
    zero_policy="detached_diagnostic",
)

COORD_GAUSSIAN_RPS_BINDING = TokenLossBinding(
    name="coord_gaussian_rps",
    role="auxiliary",
    normalizer="segment_balanced",
    zero_policy="omit",
)

#: The complete closed inventory, in canonical composition order.
TOKEN_LOSS_BINDINGS: tuple[TokenLossBinding, ...] = (
    BASE_CE_BINDING,
    TOKEN_TYPE_GATE_BINDING,
    COORD_GAUSSIAN_RPS_BINDING,
)

#: The protected base-CE weight; the `forbid` zero policy admits no other.
PROTECTED_BASE_CE_WEIGHT = 1.0

#: Token types the coordinate auxiliary selects (a code constant, never config).
COORDINATE_TOKEN_TYPES: tuple[str, ...] = ("coordinate",)

_BINDINGS_BY_NAME = {binding.name: binding for binding in TOKEN_LOSS_BINDINGS}


def binding_for(name: str) -> TokenLossBinding:
    """Return the binding for `name` from the closed inventory.

    This is an exact lookup over a compiled mapping, never a discovery or
    import-by-name mechanism: an unrecognized name is a contract error.
    """

    binding = _BINDINGS_BY_NAME.get(name)
    if binding is None:
        raise LossContractError(
            "unknown token loss name: the implemented inventory is closed",
            code="loss.unknown_token_loss_binding",
            context={"name": name, "known": list(_BINDINGS_BY_NAME)},
        )
    return binding


__all__ = [
    "BASE_CE_BINDING",
    "COORDINATE_TOKEN_TYPES",
    "COORD_GAUSSIAN_RPS_BINDING",
    "PROTECTED_BASE_CE_WEIGHT",
    "TOKEN_LOSS_BINDINGS",
    "TOKEN_TYPE_GATE_BINDING",
    "TokenLossBinding",
    "TokenLossNormalizer",
    "TokenLossRole",
    "TokenLossZeroPolicy",
    "binding_for",
]
