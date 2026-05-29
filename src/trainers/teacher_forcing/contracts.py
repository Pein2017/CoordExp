from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch

RegistryContext = Literal["gt", "rollout"]


@dataclass(frozen=True)
class PipelineModuleSpec:
    name: str
    enabled: bool = True
    weight: float = 1.0
    surfaces: tuple[str, ...] = ("rollout_correction",)
    application: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PipelineModuleSpec":
        if "channels" in payload:
            raise ValueError(
                "Pipeline module channels were removed; use the unified "
                "rollout_correction surface."
            )

        cfg_raw = payload.get("config", {})
        cfg = dict(cfg_raw) if isinstance(cfg_raw, Mapping) else {}
        application_raw = payload.get("application", {})
        application = (
            dict(application_raw) if isinstance(application_raw, Mapping) else {}
        )

        try:
            weight = float(payload.get("weight", 1.0) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0

        return cls(
            name=str(payload.get("name", "") or "").strip(),
            enabled=bool(payload.get("enabled", True)),
            weight=max(0.0, float(weight)),
            surfaces=("rollout_correction",),
            application=application,
            config=cfg,
        )

    def enabled_for_surface(self, surface: str) -> bool:
        resolved = str(surface or "").strip()
        return bool(self.enabled and resolved in set(self.surfaces))


@dataclass(frozen=True)
class TeacherForcingContext:
    channel: str
    registry_context: RegistryContext
    input_ids: torch.Tensor
    logits: torch.Tensor
    logits_ce: torch.Tensor
    meta: Sequence[Mapping[str, Any]]
    coord_token_ids: Sequence[int]
    temperature: float = 1.0
    token_type_masks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    rollout_subset_masks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def coord_id_set(self) -> set[int]:
        return {int(i) for i in self.coord_token_ids if int(i) >= 0}


@dataclass(frozen=True)
class ModuleResult:
    loss: torch.Tensor
    metrics: Mapping[str, float] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    total_loss: torch.Tensor
    module_losses: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
