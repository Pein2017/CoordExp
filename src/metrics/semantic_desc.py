"""Compatibility shim for semantic description helpers.

Canonical implementation lives in `src.common.semantic_desc` so evaluation,
training gating, and monitoring share the same normalization + encoder.
"""

from src.common.semantic_desc import (
    SemanticDescEncoder,
    normalize_desc,
    semantic_ok_and_sim,
)

__all__ = [
    "normalize_desc",
    "SemanticDescEncoder",
    "semantic_ok_and_sim",
]
