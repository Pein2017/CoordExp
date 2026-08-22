"""Adapter setup contracts for CoordExp."""

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
    build_adapter_setup_plan,
)

__all__ = [
    "DEFAULT_ADAPTER_NAME",
    "AdapterSetupPlan",
    "DoraAdapterSetupReceipt",
    "DoraAdapterSetupResult",
    "DoraTargetDiscoveryReceipt",
    "build_adapter_setup_plan",
    "discover_dora_targets",
    "setup_dora_adapter",
]
