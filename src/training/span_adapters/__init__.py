"""Compact span adapters for semantic training supervision."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.training.span_adapters.compact_projector import (
        CompactCoordinateSlotProjection,
        CompactFullSpanProjector,
        CompactObjectProjection,
        CompactSpanProjection,
    )
    from src.training.span_adapters.stage1_compact import (
        CompactCoordinateSoftTargetSpec,
        CompactCoordinateTokenWeightSpec,
        CompactTrieTargetSpec,
        Stage1CompactSpanAdapter,
    )
    from src.training.span_adapters.stage2_compact import (
        Stage2CompactSpanAdapter,
        Stage2CompactTokenTargetSpec,
        Stage2SpanProvenance,
    )

_LAZY_EXPORTS = {
    "CompactCoordinateSlotProjection": "src.training.span_adapters.compact_projector",
    "CompactFullSpanProjector": "src.training.span_adapters.compact_projector",
    "CompactObjectProjection": "src.training.span_adapters.compact_projector",
    "CompactSpanProjection": "src.training.span_adapters.compact_projector",
    "CompactCoordinateSoftTargetSpec": "src.training.span_adapters.stage1_compact",
    "CompactCoordinateTokenWeightSpec": "src.training.span_adapters.stage1_compact",
    "CompactTrieTargetSpec": "src.training.span_adapters.stage1_compact",
    "Stage1CompactSpanAdapter": "src.training.span_adapters.stage1_compact",
    "Stage2CompactSpanAdapter": "src.training.span_adapters.stage2_compact",
    "Stage2CompactTokenTargetSpec": "src.training.span_adapters.stage2_compact",
    "Stage2SpanProvenance": "src.training.span_adapters.stage2_compact",
}


def __getattr__(name: str) -> Any:
    """Resolve compact span adapter exports on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "CompactCoordinateSlotProjection",
    "CompactCoordinateSoftTargetSpec",
    "CompactCoordinateTokenWeightSpec",
    "CompactFullSpanProjector",
    "CompactObjectProjection",
    "CompactSpanProjection",
    "CompactTrieTargetSpec",
    "Stage1CompactSpanAdapter",
    "Stage2CompactSpanAdapter",
    "Stage2CompactTokenTargetSpec",
    "Stage2SpanProvenance",
]
