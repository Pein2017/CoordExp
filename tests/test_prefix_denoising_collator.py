from __future__ import annotations

import pytest

from src.data_collators.enrichers import PrefixDenoisingHybridEnricher
from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.types import HybridPrefixDenoisingSample


def _sample(sample_id: str) -> HybridPrefixDenoisingSample:
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=sample_id,
        base_sample_id=sample_id,
        clean_full=None,
        noisy_full=None,
        kl_sites=(),
    )


def test_prefix_denoising_enricher_requires_all_rows_present_unpacked() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    collated: dict[str, object] = {}

    with pytest.raises(ValueError, match="prefix_denoising_hybrid sidecar must be present"):
        enricher(
            collated=collated,
            raw_batch=[{"prefix_denoising_hybrid": _sample("a")}, {}],
            packed=False,
        )


def test_prefix_denoising_enricher_attaches_tuple_unpacked() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    collated: dict[str, object] = {}

    enricher(
        collated=collated,
        raw_batch=[{"prefix_denoising_hybrid": _sample("a")}],
        packed=False,
    )

    assert "prefix_denoising_hybrid" in collated
    assert len(collated["prefix_denoising_hybrid"]) == 1  # type: ignore[arg-type]


def test_prefix_denoising_enricher_rejects_plain_packed_sidecar_without_boundary_map() -> None:
    enricher = PrefixDenoisingHybridEnricher()

    with pytest.raises(ValueError, match="PackedHybridBoundaryMap"):
        enricher(
            collated={},
            raw_batch=[[{"prefix_denoising_hybrid": _sample("a")}]],
            packed=True,
        )


def test_prefix_denoising_enricher_attaches_packed_hybrids_when_boundary_map_exists() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    collated: dict[str, object] = {"packed_hybrid_boundary_map": object()}

    enricher(
        collated=collated,
        raw_batch=[
            [
                {"prefix_denoising_hybrid": _sample("a")},
                {"prefix_denoising_hybrid": _sample("b")},
            ]
        ],
        packed=True,
    )

    assert "prefix_denoising_hybrid" in collated
    packed_groups = collated["prefix_denoising_hybrid"]
    assert len(packed_groups) == 1  # type: ignore[arg-type]
    assert len(packed_groups[0]) == 2  # type: ignore[index]


def test_prefix_denoising_sidecars_are_registered_at_model_boundary() -> None:
    batch = {
        "input_ids": object(),
        "labels": object(),
        "prefix_denoising_hybrid": object(),
        "prefix_denoising_segment_meta": object(),
        "packed_hybrid_boundary_map": object(),
        "prefix_denoising_resolved_kl_sites": object(),
        "sample_id": "unit-0",
    }

    stripped = strip_non_model_detection_sidecars(batch)

    assert sorted(stripped) == ["input_ids", "labels"]
