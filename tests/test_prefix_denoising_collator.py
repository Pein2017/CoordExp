from __future__ import annotations

import pytest

from src.data_collators.enrichers import PrefixDenoisingHybridEnricher
from src.detection.dataset import strip_non_model_detection_sidecars
from src.detection.prefix_denoising.types import (
    HybridPrefixDenoisingSample,
    PrefixDenoisingKLSite,
    PrefixDenoisingSegment,
    ResolvedPrefixDenoisingKLSite,
)


def _sample(sample_id: str) -> HybridPrefixDenoisingSample:
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=sample_id,
        base_sample_id=sample_id,
        clean_full=None,
        noisy_full=None,
        kl_sites=(),
    )


def _hybrid_sample(sample_id: str, *, token_base: int) -> HybridPrefixDenoisingSample:
    clean = PrefixDenoisingSegment(
        segment_id=f"{sample_id}:clean",
        branch_id="clean_full",
        input_ids=(token_base, token_base + 1, token_base + 2),
        labels=(-100, -100, 1000 + token_base),
        attention_mask=(1, 1, 1),
        supervised_positions=(2,),
        ce_denominator=1,
    )
    noisy = PrefixDenoisingSegment(
        segment_id=f"{sample_id}:noisy",
        branch_id="noisy_full",
        input_ids=(token_base + 10, token_base + 11),
        labels=(-100, 1000 + token_base),
        attention_mask=(1, 1),
        supervised_positions=(1,),
        ce_denominator=1,
    )
    return HybridPrefixDenoisingSample(
        ok=True,
        hybrid_sample_id=sample_id,
        base_sample_id=f"base-{sample_id}",
        clean_full=clean,
        noisy_full=noisy,
        kl_sites=(
            PrefixDenoisingKLSite(
                clean_segment_id=clean.segment_id,
                noisy_segment_id=noisy.segment_id,
                object_index=token_base,
                history_object_count=0,
                coord_slot="x1",
                clean_label_position=2,
                noisy_label_position=1,
                clean_gt_bin=10 + token_base,
                support_bins=(9 + token_base, 10 + token_base, 11 + token_base),
                identical_prefix=True,
            ),
        ),
    )


def _row(sample_id: str, *, token_base: int) -> dict[str, object]:
    sample = _hybrid_sample(sample_id, token_base=token_base)
    assert sample.clean_full is not None
    assert sample.noisy_full is not None
    length = len(sample.clean_full.input_ids) + len(sample.noisy_full.input_ids)
    return {
        "input_ids": list(sample.clean_full.input_ids + sample.noisy_full.input_ids),
        "length": length,
        "prefix_denoising_hybrid": sample,
        "prefix_denoising_segment_meta": (
            {
                "hybrid_sample_id": sample.hybrid_sample_id,
                "segment_id": sample.clean_full.segment_id,
                "branch_id": "clean_full",
                "local_token_start": 0,
                "local_token_end": len(sample.clean_full.input_ids),
                "local_supervised_positions": sample.clean_full.supervised_positions,
            },
            {
                "hybrid_sample_id": sample.hybrid_sample_id,
                "segment_id": sample.noisy_full.segment_id,
                "branch_id": "noisy_full",
                "local_token_start": len(sample.clean_full.input_ids),
                "local_token_end": length,
                "local_supervised_positions": sample.noisy_full.supervised_positions,
            },
        ),
    }


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


def test_prefix_denoising_enricher_rejects_unpacked_companion_without_hybrid() -> None:
    enricher = PrefixDenoisingHybridEnricher()

    with pytest.raises(ValueError, match="prefix_denoising_segment_meta.*without"):
        enricher(
            collated={},
            raw_batch=[{"prefix_denoising_segment_meta": object()}],
            packed=False,
        )


def test_prefix_denoising_enricher_builds_packed_boundary_map_from_sample_lengths() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    first = _row("a", token_base=1)
    second = _row("b", token_base=2)
    collated: dict[str, object] = {}

    enricher(
        collated=collated,
        raw_batch=[[first, second]],
        packed=True,
    )

    assert collated["packed_hybrid_boundary_map"] == (
        (
            {"sample_index": 0, "start": 0, "end": 5},
            {"sample_index": 1, "start": 5, "end": 10},
        ),
    )


def test_prefix_denoising_enricher_rejects_packed_companion_without_hybrid() -> None:
    enricher = PrefixDenoisingHybridEnricher()

    with pytest.raises(ValueError, match="prefix_denoising_resolved_kl_sites.*without"):
        enricher(
            collated={},
            raw_batch=[[{"prefix_denoising_resolved_kl_sites": object()}]],
            packed=True,
        )


def test_prefix_denoising_enricher_attaches_packed_hybrids_when_boundary_map_exists() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    collated: dict[str, object] = {}

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


def test_prefix_denoising_enricher_globalizes_packed_segments_and_kl_sites() -> None:
    enricher = PrefixDenoisingHybridEnricher()
    first = _row("a", token_base=1)
    second = _row("b", token_base=2)
    collated: dict[str, object] = {}

    enricher(
        collated=collated,
        raw_batch=[[first, second]],
        packed=True,
    )

    segment_meta = collated["prefix_denoising_segment_meta"]
    assert segment_meta[0][0][0]["local_token_start"] == 0  # type: ignore[index]
    assert segment_meta[0][0][0]["local_token_end"] == 3  # type: ignore[index]
    assert segment_meta[0][0][1]["local_token_start"] == 3  # type: ignore[index]
    assert segment_meta[0][0][1]["local_token_end"] == 5  # type: ignore[index]
    assert segment_meta[0][1][0]["local_token_start"] == 5  # type: ignore[index]
    assert segment_meta[0][1][0]["local_token_end"] == 8  # type: ignore[index]
    assert segment_meta[0][1][1]["local_token_start"] == 8  # type: ignore[index]
    assert segment_meta[0][1][1]["local_token_end"] == 10  # type: ignore[index]
    assert segment_meta[0][1][0]["local_supervised_positions"] == (2,)  # type: ignore[index]
    assert segment_meta[0][1][1]["local_supervised_positions"] == (1,)  # type: ignore[index]
    assert segment_meta[0][1][0]["batch_index"] == 0  # type: ignore[index]

    resolved = collated["prefix_denoising_resolved_kl_sites"]
    assert isinstance(resolved[0][1][0], ResolvedPrefixDenoisingKLSite)  # type: ignore[index]
    second_site = resolved[0][1][0]  # type: ignore[index]
    assert second_site.clean_batch_index == 0
    assert second_site.noisy_batch_index == 0
    assert second_site.clean_label_position == 7
    assert second_site.noisy_label_position == 9
    assert second_site.support_bins == (11, 12, 13)
    assert second_site.clean_gt_bin == 12
    assert second_site.coord_slot == "x1"
    assert second_site.object_index == 2
    assert second_site.identical_prefix is True


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
