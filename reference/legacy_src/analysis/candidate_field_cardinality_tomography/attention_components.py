from __future__ import annotations

from typing import Mapping


REQUIRED_ATTENTION_COMPONENT_FIELDS = (
    "role",
    "layer",
    "head",
    "aggregation_scope",
    "region_kind",
    "target_region_mass",
    "same_desc_competitor_region_mass",
    "background_region_mass",
    "sink_or_special_token_mass",
    "attention_aggregation_policy_id",
)


def normalize_region_masses(row: Mapping[str, float]) -> dict[str, float]:
    keys = ("target_region_mass", "same_desc_competitor_region_mass", "background_region_mass", "sink_or_special_token_mass")
    total = sum(float(row.get(key, 0.0)) for key in keys)
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {key: float(row.get(key, 0.0)) / total for key in keys}


def validate_attention_component_row(row: Mapping[str, object]) -> None:
    for field in REQUIRED_ATTENTION_COMPONENT_FIELDS:
        if field not in row:
            raise ValueError(f"missing attention component field: {field}")
    for field in (
        "target_region_mass",
        "same_desc_competitor_region_mass",
        "background_region_mass",
        "sink_or_special_token_mass",
    ):
        mass = float(row[field])
        if mass < 0:
            raise ValueError(f"negative attention mass: {field}")


def attention_only_taxonomy_bucket() -> str:
    return "attention_only_unassigned"
