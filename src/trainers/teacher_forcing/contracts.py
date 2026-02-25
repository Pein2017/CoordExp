from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Sequence

import torch

RegistryContext = Literal["gt", "self_context", "rollout"]


@dataclass(frozen=True)
class PipelineModuleSpec:
    name: str
    enabled: bool = True
    weight: float = 1.0
    channels: tuple[str, ...] = ("A", "B")
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PipelineModuleSpec":
        channels_raw = payload.get("channels", ("A", "B"))
        channels: list[str] = []
        if isinstance(channels_raw, Sequence) and not isinstance(channels_raw, (str, bytes)):
            for ch in channels_raw:
                ch_s = str(ch).strip().upper()
                if ch_s in {"A", "B"}:
                    channels.append(ch_s)
        if not channels:
            channels = ["A", "B"]

        cfg_raw = payload.get("config", {})
        cfg = dict(cfg_raw) if isinstance(cfg_raw, Mapping) else {}

        try:
            weight = float(payload.get("weight", 1.0) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0

        return cls(
            name=str(payload.get("name", "") or "").strip(),
            enabled=bool(payload.get("enabled", True)),
            weight=max(0.0, float(weight)),
            channels=tuple(channels),
            config=cfg,
        )

    def enabled_for_channel(self, channel: str) -> bool:
        ch = str(channel or "").strip().upper()
        return bool(self.enabled and ch in set(self.channels))


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
    decode_mode: str = "exp"
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
