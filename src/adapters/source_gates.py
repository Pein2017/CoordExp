"""Direct, config-bound DoRA setup plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.common.errors import RuntimeContractError
from src.config.models import AdapterConfig


AdapterSetupMode = Literal["initialize_new", "load_existing", "warm_start_expand_dora"]


@dataclass(frozen=True)
class AdapterSetupPlan:
    mode: AdapterSetupMode
    adapter_type: Literal["dora"]
    adapter_path: Path | None
    source_adapter_path: Path | None
    repaired_embedding_payload_path: Path | None
    base_model_path: Path | None
    target_towers: tuple[str, ...]
    target_policy: Literal["all_linear"]
    rank: int
    alpha: int
    dropout: float
    bias: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "adapter_type": self.adapter_type,
            "adapter_identity": None
            if self.adapter_path is None
            else {"path": str(self.adapter_path)},
            "source_adapter_identity": None
            if self.source_adapter_path is None
            else {"path": str(self.source_adapter_path)},
            "repaired_embedding_payload_identity": None
            if self.repaired_embedding_payload_path is None
            else {"path": str(self.repaired_embedding_payload_path)},
            "base_model_identity": None
            if self.base_model_path is None
            else {"path": str(self.base_model_path)},
            "target_towers": list(self.target_towers),
            "target_policy": self.target_policy,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
        }


def build_adapter_setup_plan(
    adapter_config: AdapterConfig,
    *,
    base_model_path: str | Path | None = None,
) -> AdapterSetupPlan:
    if adapter_config.type != "dora":
        raise RuntimeContractError(
            "unsupported V1 adapter type; V1 uses adapter.type: dora",
            code="adapter.unsupported_type",
            context={"adapter_type": adapter_config.type},
        )
    adapter_path = None if adapter_config.path is None else Path(adapter_config.path)
    source_adapter_path = (
        None
        if adapter_config.source_adapter_path is None
        else Path(adapter_config.source_adapter_path)
    )
    repaired_embedding_payload_path = (
        None
        if adapter_config.repaired_embedding_payload_path is None
        else Path(adapter_config.repaired_embedding_payload_path)
    )
    resolved_base_model_path = (
        None if base_model_path is None else Path(base_model_path)
    )
    seed_mode = adapter_config.seed_mode
    mode: AdapterSetupMode = (
        ("initialize_new" if adapter_path is None else "load_existing")
        if seed_mode is None
        else seed_mode
    )
    if mode == "warm_start_expand_dora" and (
        source_adapter_path is None or repaired_embedding_payload_path is None
    ):
        raise RuntimeContractError(
            "warm_start_expand_dora requires source adapter and repaired embedding payload paths",
            code="adapter.warm_start_source_paths_required",
            context={
                "source_adapter_path": None
                if source_adapter_path is None
                else str(source_adapter_path),
                "repaired_embedding_payload_path": None
                if repaired_embedding_payload_path is None
                else str(repaired_embedding_payload_path),
            },
        )
    if (
        mode in {"load_existing", "warm_start_expand_dora"}
        and resolved_base_model_path is None
    ):
        raise RuntimeContractError(
            "adapter setup must record base model identity",
            code="adapter.base_model_identity_required",
            context={
                "mode": mode,
                "adapter_path": None if adapter_path is None else str(adapter_path),
                "source_adapter_path": None
                if source_adapter_path is None
                else str(source_adapter_path),
            },
        )
    return AdapterSetupPlan(
        mode=mode,
        adapter_type="dora",
        adapter_path=adapter_path,
        source_adapter_path=source_adapter_path,
        repaired_embedding_payload_path=repaired_embedding_payload_path,
        base_model_path=resolved_base_model_path,
        target_towers=tuple(adapter_config.target_towers),
        target_policy=adapter_config.target_modules,
        rank=adapter_config.rank,
        alpha=adapter_config.alpha,
        dropout=adapter_config.dropout,
        bias=adapter_config.bias,
    )


__all__ = ["AdapterSetupPlan", "build_adapter_setup_plan"]
