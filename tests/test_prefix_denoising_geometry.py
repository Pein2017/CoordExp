from __future__ import annotations

import random

import pytest

from src.datasets.geometry import (
    BBoxNoiseConfig,
    construct_valid_norm1000_bbox_noise,
)


def _cfg() -> BBoxNoiseConfig:
    return BBoxNoiseConfig(center_shift_frac=0.08, uniform_scale_range=(0.92, 1.08))


def test_constructive_noise_returns_valid_4coord_changed_bbox() -> None:
    result = construct_valid_norm1000_bbox_noise(
        (100, 120, 260, 300),
        config=_cfg(),
        rng=random.Random(7),
        object_id="obj-1",
    )

    assert result.ok is True
    assert result.noisy_bbox is not None
    x1, y1, x2, y2 = result.noisy_bbox
    assert 0 <= x1 < x2 <= 999
    assert 0 <= y1 < y2 <= 999
    assert tuple(result.changed) == (True, True, True, True)
    assert result.clean_bins == (100, 120, 260, 300)
    assert result.noisy_bins != result.clean_bins


def test_constructive_noise_is_seed_deterministic() -> None:
    a = construct_valid_norm1000_bbox_noise(
        (100, 120, 260, 300), config=_cfg(), rng=random.Random(11)
    )
    b = construct_valid_norm1000_bbox_noise(
        (100, 120, 260, 300), config=_cfg(), rng=random.Random(11)
    )

    assert a.noisy_bbox == b.noisy_bbox
    assert a.provenance == b.provenance


def test_constructive_noise_respects_scale_range_excluding_one() -> None:
    result = construct_valid_norm1000_bbox_noise(
        (100, 120, 300, 420),
        config=BBoxNoiseConfig(center_shift_frac=0.08, uniform_scale_range=(0.5, 0.6)),
        rng=random.Random(1),
    )

    assert result.ok is True
    assert result.noisy_bbox is not None
    nx1, ny1, nx2, ny2 = result.noisy_bbox
    noisy_width = nx2 - nx1
    noisy_height = ny2 - ny1
    assert round(200 * 0.5) <= noisy_width <= round(200 * 0.6)
    assert round(300 * 0.5) <= noisy_height <= round(300 * 0.6)


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 1, 1),
        (0, 0, 999, 999),
        (998, 998, 999, 999),
        (10, 10, 11, 500),
        (10, 10, 500, 11),
    ],
)
def test_constructive_noise_reports_infeasible_without_repair(
    bbox: tuple[int, int, int, int],
) -> None:
    result = construct_valid_norm1000_bbox_noise(
        bbox,
        config=BBoxNoiseConfig(
            center_shift_frac=0.0, uniform_scale_range=(1.0, 1.0)
        ),
        rng=random.Random(3),
    )

    assert result.ok is False
    assert result.skip_reason in {
        "noise_infeasible_valid_bbox",
        "noise_infeasible_4coord_changed",
    }
    assert result.noisy_bbox is None


def test_invalid_clean_bbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="clean bbox"):
        construct_valid_norm1000_bbox_noise(
            (5, 5, 5, 6), config=_cfg(), rng=random.Random(1)
        )


@pytest.mark.parametrize("bad_value", ["x", float("nan")])
def test_invalid_clean_bbox_scalar_errors_include_field_name(bad_value: object) -> None:
    with pytest.raises(ValueError, match="clean bbox"):
        construct_valid_norm1000_bbox_noise(
            (bad_value, 5, 6, 7), config=_cfg(), rng=random.Random(1)
        )
