from __future__ import annotations

from dataclasses import replace

import pytest

from src.detection.packing import (
    PackingProfile,
    assess_packing_eligibility,
    require_packing_eligibility,
)
from src.detection.template import Stage1JsonPrettyTemplate


class StaticPackingUnsupportedTemplate(Stage1JsonPrettyTemplate):
    capabilities = replace(
        Stage1JsonPrettyTemplate.capabilities,
        supports_static_packing=False,
    )


def _static_profile() -> PackingProfile:
    return PackingProfile(mode="static", packing_length=2048)


def _padding_free_profile() -> PackingProfile:
    return PackingProfile(
        mode="padding_free_packed",
        packing_length=2048,
        runtime_flags={
            "attention_backend": "flash_attention_2",
            "lossless_boundaries": True,
        },
    )


@pytest.mark.parametrize("training_mode", ["sorted_sft", "random_order_sft"])
def test_static_packing_is_eligible_for_full_sequence_sft_examples(
    training_mode: str,
) -> None:
    eligibility = assess_packing_eligibility(
        template=Stage1JsonPrettyTemplate(),
        training_mode=training_mode,
        profile=_static_profile(),
    )

    assert eligibility.eligible is True
    assert eligibility.mode == "static"
    assert eligibility.training_mode == training_mode
    assert eligibility.experimental is False
    assert "full-sequence hard-CE" in eligibility.reason


def test_recursive_detection_ce_static_packing_is_rejected_until_trie_metadata_is_preserved() -> None:
    profile = _static_profile()
    eligibility = assess_packing_eligibility(
        template=Stage1JsonPrettyTemplate(),
        training_mode="random_permutation_et_rmp_ce",
        profile=profile,
    )

    assert eligibility.eligible is False
    assert eligibility.requires_trie_metadata_preservation is True
    assert "trie target metadata preservation" in eligibility.reason

    with pytest.raises(ValueError, match="trie target metadata preservation"):
        require_packing_eligibility(
            template=Stage1JsonPrettyTemplate(),
            training_mode="random_permutation_et_rmp_ce",
            profile=profile,
        )


def test_static_packing_rejects_templates_without_static_capability() -> None:
    eligibility = assess_packing_eligibility(
        template=StaticPackingUnsupportedTemplate(),
        training_mode="sorted_sft",
        profile=_static_profile(),
    )

    assert eligibility.eligible is False
    assert "does not support static packing" in eligibility.reason

    with pytest.raises(ValueError, match="does not support static packing"):
        require_packing_eligibility(
            template=StaticPackingUnsupportedTemplate(),
            training_mode="sorted_sft",
            profile=_static_profile(),
        )


def test_padding_free_mode_is_explicit_and_experimental() -> None:
    profile = _padding_free_profile()
    eligibility = assess_packing_eligibility(
        template=Stage1JsonPrettyTemplate(),
        training_mode="sorted_sft",
        profile=profile,
    )

    assert profile.experimental is True
    assert eligibility.eligible is False
    assert eligibility.experimental is True
    assert "experimental" in eligibility.reason
    assert "static packing" in eligibility.reason

    with pytest.raises(ValueError, match="experimental"):
        require_packing_eligibility(
            template=Stage1JsonPrettyTemplate(),
            training_mode="sorted_sft",
            profile=profile,
        )
