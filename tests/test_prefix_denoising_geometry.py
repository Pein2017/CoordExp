from __future__ import annotations

import random

import pytest

from src.datasets.geometry import (
    BBoxNoiseConfig,
    construct_valid_norm1000_bbox_noise,
    valid_norm1000_bbox_noise_candidate_count,
)


def _cfg() -> BBoxNoiseConfig:
    return BBoxNoiseConfig(center_shift_frac=0.08, uniform_scale_range=(0.92, 1.08))


def _brute_force_candidate_count(
    bbox: tuple[int, int, int, int],
    *,
    config: BBoxNoiseConfig,
) -> int:
    return len(_brute_force_candidates(bbox, config=config))


def _brute_force_candidates(
    bbox: tuple[int, int, int, int],
    *,
    config: BBoxNoiseConfig,
) -> tuple[tuple[int, int, int, int], ...]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    max_dx = int(round(width * float(config.center_shift_frac)))
    max_dy = int(round(height * float(config.center_shift_frac)))
    scale_low, scale_high = config.uniform_scale_range
    scale_values = {scale_low, scale_high}
    if scale_low <= 1.0 <= scale_high:
        scale_values.add(1.0)
    cx2 = x1 + x2
    cy2 = y1 + y2
    candidates: set[tuple[int, int, int, int]] = set()
    for sx in sorted(scale_values):
        new_w = int(round(width * sx))
        if new_w <= 0:
            continue
        for sy in sorted(scale_values):
            new_h = int(round(height * sy))
            if new_h <= 0:
                continue
            for dx in range(-max_dx, max_dx + 1):
                for dy in range(-max_dy, max_dy + 1):
                    new_cx2 = cx2 + 2 * dx
                    new_cy2 = cy2 + 2 * dy
                    nx1 = int(round((new_cx2 - new_w) / 2.0))
                    ny1 = int(round((new_cy2 - new_h) / 2.0))
                    nx2 = nx1 + new_w
                    ny2 = ny1 + new_h
                    candidate = (nx1, ny1, nx2, ny2)
                    if not (0 <= nx1 < nx2 <= 999 and 0 <= ny1 < ny2 <= 999):
                        continue
                    if all(a != b for a, b in zip(bbox, candidate, strict=True)):
                        candidates.add(candidate)
    return tuple(sorted(candidates))


@pytest.mark.parametrize(
    "bbox",
    [
        (100, 120, 260, 300),
        (0, 0, 40, 50),
        (930, 920, 999, 999),
    ],
)
def test_fast_bbox_noise_candidate_count_matches_bruteforce(
    bbox: tuple[int, int, int, int],
) -> None:
    config = _cfg()

    assert valid_norm1000_bbox_noise_candidate_count(bbox, config=config) == (
        _brute_force_candidate_count(bbox, config=config)
    )


@pytest.mark.parametrize(
    "bbox",
    [
        (100, 120, 260, 300),
        (0, 0, 40, 50),
        (930, 920, 999, 999),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 7, 11, 123])
def test_fast_bbox_noise_selection_matches_bruteforce_sorted_candidate_order(
    bbox: tuple[int, int, int, int],
    seed: int,
) -> None:
    config = _cfg()
    candidates = _brute_force_candidates(bbox, config=config)
    assert candidates

    result = construct_valid_norm1000_bbox_noise(
        bbox,
        config=config,
        rng=random.Random(seed),
    )
    selected_index = random.Random(seed).randrange(len(candidates))

    assert result.provenance["candidate_count"] == len(candidates)
    assert result.provenance["selected_index"] == selected_index
    assert result.noisy_bbox == candidates[selected_index]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"center_shift_frac": -0.01}, "BBoxNoiseConfig.center_shift_frac"),
        ({"center_shift_frac": float("nan")}, "BBoxNoiseConfig.center_shift_frac"),
        ({"center_shift_frac": True}, "BBoxNoiseConfig.center_shift_frac"),
        ({"uniform_scale_range": (1.08, 0.92)}, "BBoxNoiseConfig.uniform_scale_range"),
        ({"uniform_scale_range": (0.9,)}, "BBoxNoiseConfig.uniform_scale_range"),
        (
            {"uniform_scale_range": (0.9, float("inf"))},
            "BBoxNoiseConfig.uniform_scale_range",
        ),
        ({"uniform_scale_range": (0.9, False)}, "BBoxNoiseConfig.uniform_scale_range"),
        ({"coord_min": -1}, "BBoxNoiseConfig.coord_min"),
        ({"coord_max": 1000}, "BBoxNoiseConfig.coord_max"),
    ],
)
def test_bbox_noise_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BBoxNoiseConfig(**kwargs)


def test_bbox_noise_config_normalizes_scale_range_to_float_tuple() -> None:
    config = BBoxNoiseConfig(uniform_scale_range=[1, 2])

    assert config.uniform_scale_range == (1.0, 2.0)
    assert all(isinstance(value, float) for value in config.uniform_scale_range)


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
    assert result.skip_reason == "noise_infeasible_4coord_changed"
    assert result.noisy_bbox is None


def test_constructive_noise_skips_zero_size_scaled_candidates_without_repair() -> None:
    result = construct_valid_norm1000_bbox_noise(
        (10, 10, 11, 11),
        config=BBoxNoiseConfig(
            center_shift_frac=1.0, uniform_scale_range=(0.4, 0.49)
        ),
        rng=random.Random(9),
    )

    assert result.ok is False
    assert result.noisy_bbox is None
    assert result.skip_reason == "noise_infeasible_4coord_changed"


def test_invalid_clean_bbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="clean bbox"):
        construct_valid_norm1000_bbox_noise(
            (5, 5, 5, 6), config=_cfg(), rng=random.Random(1)
        )


@pytest.mark.parametrize(
    "bad_bbox", [None, 5, "1234", b"1234", {0: 0, 1: 1, 2: 2, 3: 3}]
)
def test_invalid_clean_bbox_container_errors_include_field_name(bad_bbox: object) -> None:
    with pytest.raises(ValueError, match="clean bbox"):
        construct_valid_norm1000_bbox_noise(
            bad_bbox, config=_cfg(), rng=random.Random(1)
        )


@pytest.mark.parametrize("bad_value", ["x", float("nan")])
def test_invalid_clean_bbox_scalar_errors_include_field_name(bad_value: object) -> None:
    with pytest.raises(ValueError, match="clean bbox"):
        construct_valid_norm1000_bbox_noise(
            (bad_value, 5, 6, 7), config=_cfg(), rng=random.Random(1)
        )


def test_border_touching_large_bbox_high_shift_smoke_is_bounded() -> None:
    result = construct_valid_norm1000_bbox_noise(
        (0, 0, 900, 900),
        config=BBoxNoiseConfig(center_shift_frac=1.0, uniform_scale_range=(1.0, 1.0)),
        rng=random.Random(13),
    )

    assert result.ok is True
    assert result.provenance["candidate_count"] < 500_000
    assert result.noisy_bbox is not None
    x1, y1, x2, y2 = result.noisy_bbox
    assert 0 <= x1 < x2 <= 999
    assert 0 <= y1 < y2 <= 999
