"""Adapter setup contracts for CoordExp-swift."""

from src.adapters.dora import (
    DEFAULT_ADAPTER_NAME,
    DoraAdapterSetupReceipt,
    DoraAdapterSetupResult,
    DoraTargetDiscoveryReceipt,
    discover_dora_targets,
    setup_dora_adapter,
)
from src.adapters.source_gates import (
    AdapterSetupPlan,
    AdapterSourceGateEvidence,
    DoraSourceGateReceipt,
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
    load_dora_probe_receipt,
)

__all__ = [
    "DEFAULT_ADAPTER_NAME",
    "AdapterSetupPlan",
    "AdapterSourceGateEvidence",
    "DoraAdapterSetupReceipt",
    "DoraAdapterSetupResult",
    "DoraSourceGateReceipt",
    "DoraTargetDiscoveryReceipt",
    "build_adapter_setup_plan",
    "discover_dora_targets",
    "load_default_adapter_source_gate_evidence",
    "load_dora_probe_receipt",
    "setup_dora_adapter",
]
