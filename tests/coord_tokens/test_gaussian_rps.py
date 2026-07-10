import torch
import pytest

from src.coord_tokens.gaussian_rps import (
    gaussian_soft_targets_from_r95,
    ranked_probability_score,
)
from src.trainers.losses.coord_gaussian_rps import infer_shape_aware_r95_radii


def _one_hot(index: int, *, bins: int = 1000) -> torch.Tensor:
    out = torch.zeros((1, int(bins)), dtype=torch.float32)
    out[0, int(index)] = 1.0
    return out


def test_ranked_probability_score_zero_for_identical_distributions() -> None:
    probs = _one_hot(123)

    rps = ranked_probability_score(probs, probs, normalize=True)

    assert rps.shape == (1,)
    assert float(rps.item()) == 0.0


def test_ranked_probability_score_increases_with_ordered_distance() -> None:
    target = _one_hot(500)
    near = _one_hot(502)
    far = _one_hot(600)

    near_rps = ranked_probability_score(near, target, normalize=True)
    far_rps = ranked_probability_score(far, target, normalize=True)

    assert float(far_rps.item()) > float(near_rps.item()) > 0.0


def test_gaussian_soft_targets_from_r95_are_gt_centered_and_normalized() -> None:
    target = torch.tensor([500], dtype=torch.long)
    probs = gaussian_soft_targets_from_r95(
        target,
        torch.tensor([4], dtype=torch.long),
        num_bins=1000,
    )

    assert probs.shape == (1, 1000)
    assert torch.all(probs >= 0).item()
    assert float(probs.sum().item()) == pytest.approx(1.0)
    assert int(probs.argmax(dim=-1).item()) == 500


def test_shape_aware_r95_uses_bbox_axis_length_for_xyxy_quads() -> None:
    # Two compact-full xyxy quads in one adjacent coord-token run.
    # First box: width=20, height=60.
    # Second box: width=800, height=800.
    target_bins_all = torch.tensor(
        [[10, 20, 30, 80, 100, 110, 900, 910]],
        dtype=torch.long,
    )
    labels_next = target_bins_all.clone()
    coord_positions_mask = torch.ones_like(target_bins_all, dtype=torch.bool)

    radii = infer_shape_aware_r95_radii(
        labels_next=labels_next,
        target_bins_all=target_bins_all,
        coord_positions_mask=coord_positions_mask,
        gaussian_r95_axis_fraction=0.10,
        gaussian_r95_cap_bins=99,
        gaussian_r95_min_bins=1,
        gaussian_r95_fallback_bins=8,
    )

    assert radii.tolist() == [2, 6, 2, 6, 80, 80, 80, 80]
    assert int(radii[4].item()) > int(radii[0].item())
    assert int(radii[5].item()) > int(radii[1].item())
