from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyProbeSpec:
    alias: str
    native_slots: tuple[str, ...]
    requires_family_native_probe: bool
    requires_canonical_projection: bool


_REGISTRY: dict[str, FamilyProbeSpec] = {
    "base_xyxy_merged": FamilyProbeSpec(
        alias="base_xyxy_merged",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "raw_text_xyxy_pure_ce": FamilyProbeSpec(
        alias="raw_text_xyxy_pure_ce",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "cxcywh_pure_ce": FamilyProbeSpec(
        alias="cxcywh_pure_ce",
        native_slots=("cx", "cy", "w", "h"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "cxcy_logw_logh_pure_ce": FamilyProbeSpec(
        alias="cxcy_logw_logh_pure_ce",
        native_slots=("cx", "cy", "logw", "logh"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "center_parameterization": FamilyProbeSpec(
        alias="center_parameterization",
        native_slots=("cx", "cy", "w", "h"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
    "hard_soft_ce_2b": FamilyProbeSpec(
        alias="hard_soft_ce_2b",
        native_slots=("x1", "y1", "x2", "y2"),
        requires_family_native_probe=True,
        requires_canonical_projection=True,
    ),
}


def get_family_probe_spec(alias: str) -> FamilyProbeSpec:
    return _REGISTRY[alias]


def native_slot_names(alias: str) -> tuple[str, ...]:
    return get_family_probe_spec(alias).native_slots


__all__ = [
    "FamilyProbeSpec",
    "get_family_probe_spec",
    "native_slot_names",
]
