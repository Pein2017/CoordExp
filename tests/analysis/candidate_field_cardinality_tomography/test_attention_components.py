from __future__ import annotations

from src.analysis.candidate_field_cardinality_tomography.attention_components import (
    attention_only_taxonomy_bucket,
    normalize_region_masses,
    validate_attention_component_row,
)


def test_region_masses_normalize_background_and_sink_components() -> None:
    masses = normalize_region_masses(
        {
            "target_region_mass": 2.0,
            "same_desc_competitor_region_mass": 1.0,
            "background_region_mass": 1.0,
            "sink_or_special_token_mass": 2.0,
        }
    )

    assert masses["target_region_mass"] == 2.0 / 6.0
    assert masses["background_region_mass"] == 1.0 / 6.0
    assert masses["sink_or_special_token_mass"] == 2.0 / 6.0


def test_attention_component_row_requires_policy_and_region_fields() -> None:
    row = {
        "role": "pre_x1",
        "layer": 10,
        "head": 3,
        "aggregation_scope": "vision_tokens",
        "region_kind": "target_vs_background",
        "target_region_mass": 0.2,
        "same_desc_competitor_region_mass": 0.1,
        "background_region_mass": 0.6,
        "sink_or_special_token_mass": 0.1,
        "attention_aggregation_policy_id": "region_mass_v1",
    }

    validate_attention_component_row(row)


def test_attention_only_evidence_cannot_assign_mechanism_bucket() -> None:
    assert attention_only_taxonomy_bucket() == "attention_only_unassigned"
