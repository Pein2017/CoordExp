from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.detection.coord_soft_targets import (
    CoordSoftTargetCandidate,
    CoordSoftTargetRuntimeConfig,
    build_iou_gibbs_coord_target,
    full_vocab_coord_support_balance_ce,
)


def _cfg() -> CoordSoftTargetRuntimeConfig:
    return CoordSoftTargetRuntimeConfig(
        target_distribution="iou_gibbs_v0",
        tau=0.0090909091,
        coord_token_start=10,
        coord_token_end=1009,
    )


def test_iou_gibbs_target_is_normalized_and_peaks_at_gt_coord() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (100, 100, 200, 200), 1.0)

    target = build_iou_gibbs_coord_target((candidate,), _cfg())

    assert target.token_ids.shape == (1000,)
    assert target.probs.sum().item() == pytest.approx(1.0)
    peak_index = int(target.probs.argmax().item())
    assert int(target.token_ids[peak_index].item()) == 110
    assert target.entropy.item() > 0.0


def test_boundary_invalid_candidates_get_zero_mass() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (0, 100, 5, 200), 1.0)

    target = build_iou_gibbs_coord_target((candidate,), _cfg())

    invalid = target.token_ids >= 15
    assert torch.all(target.probs[invalid] == 0)
    assert target.probs.sum().item() == pytest.approx(1.0)


def test_large_box_has_broader_target_than_tiny_box() -> None:
    tiny = CoordSoftTargetCandidate("tiny", "x1", (100, 100, 110, 200), 1.0)
    large = CoordSoftTargetCandidate("large", "x1", (100, 100, 700, 200), 1.0)

    tiny_target = build_iou_gibbs_coord_target((tiny,), _cfg())
    large_target = build_iou_gibbs_coord_target((large,), _cfg())

    assert large_target.std.item() > tiny_target.std.item()
    assert large_target.entropy.item() > tiny_target.entropy.item()


def test_support2_balance1_differs_from_pure_softce_but_preserves_full_vocab_pressure() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (3, 1, 8, 9), 1.0)
    logits = torch.zeros((1020,), dtype=torch.float32)
    logits[0] = 5.0

    result = full_vocab_coord_support_balance_ce(
        logits,
        (candidate,),
        _cfg(),
        support_weight=2.0,
        balance_weight=1.0,
    )

    assert result.weighted_loss.item() > result.pure_soft_ce_equiv.item()
    assert result.pure_soft_ce_equiv.item() > 5.0


def test_support_balance_11_equals_manual_full_vocab_softce() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (100, 100, 200, 200), 1.0)
    logits = torch.linspace(-2.0, 2.0, steps=1020, dtype=torch.float32)
    cfg = _cfg()
    dist = build_iou_gibbs_coord_target((candidate,), cfg)

    result = full_vocab_coord_support_balance_ce(
        logits,
        (candidate,),
        cfg,
        support_weight=1.0,
        balance_weight=1.0,
    )
    manual = -(
        dist.probs * F.log_softmax(logits, dim=-1).index_select(0, dist.token_ids)
    ).sum()

    assert result.weighted_loss.item() == pytest.approx(manual.item())


def test_geometry_valid_support_mask_does_not_depend_on_probability_underflow() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (100, 100, 900, 900), 1.0)
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="iou_gibbs_v0",
        tau=1e-6,
        coord_token_start=10,
        coord_token_end=1009,
    )

    dist = build_iou_gibbs_coord_target((candidate,), cfg)

    assert int(dist.support_mask.sum().item()) == 900
    assert int((dist.probs > 0).sum().item()) <= int(dist.support_mask.sum().item())


def test_iou_gibbs_target_rejects_mixed_candidate_slots() -> None:
    candidates = (
        CoordSoftTargetCandidate("box-x", "x1", (100, 100, 200, 200), 0.5),
        CoordSoftTargetCandidate("box-y", "y1", (100, 100, 200, 200), 0.5),
    )

    with pytest.raises(ValueError, match="same coordinate slot"):
        build_iou_gibbs_coord_target(candidates, _cfg())


def test_nonfinite_logits_raise_instead_of_nan_to_num() -> None:
    candidate = CoordSoftTargetCandidate("box", "x1", (3, 1, 8, 9), 1.0)
    logits = torch.zeros((1020,), dtype=torch.float32)
    logits[11] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        full_vocab_coord_support_balance_ce(
            logits,
            (candidate,),
            _cfg(),
            support_weight=2.0,
            balance_weight=1.0,
        )
