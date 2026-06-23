"""Coverage-ledger auxiliary-loss contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "CoverageLedgerObjectEntry",
    "CoverageLedgerSidecar",
    "build_coverage_ledger_sidecar",
]


if TYPE_CHECKING:
    from src.training.coverage_ledger.sidecar_builder import (
        build_coverage_ledger_sidecar,
    )
    from src.training.coverage_ledger.sidecars import (
        CoverageLedgerObjectEntry,
        CoverageLedgerSidecar,
    )


def __getattr__(name: str) -> Any:
    if name == "build_coverage_ledger_sidecar":
        from src.training.coverage_ledger.sidecar_builder import (
            build_coverage_ledger_sidecar,
        )

        return build_coverage_ledger_sidecar
    if name in {"CoverageLedgerObjectEntry", "CoverageLedgerSidecar"}:
        from src.training.coverage_ledger.sidecars import (
            CoverageLedgerObjectEntry,
            CoverageLedgerSidecar,
        )

        return {
            "CoverageLedgerObjectEntry": CoverageLedgerObjectEntry,
            "CoverageLedgerSidecar": CoverageLedgerSidecar,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
