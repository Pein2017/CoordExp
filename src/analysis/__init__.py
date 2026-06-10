"""Offline analysis helpers for CoordExp research studies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = ["main", "run_rollout_fn_factor_study"]


if TYPE_CHECKING:
    from .rollout_fn_factor_study import run_rollout_fn_factor_study
    from .unmatched_proposal_verifier import main


def __getattr__(name: str) -> Any:
    if name == "main":
        from .unmatched_proposal_verifier import main

        return main
    if name == "run_rollout_fn_factor_study":
        from .rollout_fn_factor_study import run_rollout_fn_factor_study

        return run_rollout_fn_factor_study
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
