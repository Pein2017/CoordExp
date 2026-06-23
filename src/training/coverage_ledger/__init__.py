"""Coverage-ledger auxiliary-loss sidecar contracts."""

from src.training.coverage_ledger.sidecar_builder import build_coverage_ledger_sidecar
from src.training.coverage_ledger.sidecars import (
    CoverageLedgerObjectEntry,
    CoverageLedgerSidecar,
)

__all__ = [
    "CoverageLedgerObjectEntry",
    "CoverageLedgerSidecar",
    "build_coverage_ledger_sidecar",
]
