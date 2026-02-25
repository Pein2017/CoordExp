from __future__ import annotations

import pytest

from src.trainers.stage2_two_channel import _PendingStage2Log


def test_stage2_pending_log_finalize_averages_losses_and_sums_counters() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 1.0,
            "loss/B_coord/coord_soft_ce": 2.0,
            "stage2/channel_b": 1.0,
            "stage2/raw_rollouts": 1.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 1.0,
        }
    )
    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 3.0,
            "loss/B_coord/coord_soft_ce": 4.0,
            "stage2/channel_b": 0.0,
            "stage2/raw_rollouts": 2.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 0.0,
        }
    )

    out = pending.finalize(drop_internal=False)

    # Averaged across micro-batches (n_micro=2).
    assert out["loss/B_coord/bbox_smoothl1"] == pytest.approx(2.0)
    assert out["loss/B_coord/coord_soft_ce"] == pytest.approx(3.0)
    assert out["stage2/channel_b"] == pytest.approx(0.5)
    assert out["rollout/seed_base"] == pytest.approx(10.0)

    # Summed counters.
    assert out["stage2/raw_rollouts"] == pytest.approx(3.0)
    assert out["rollout/parse_truncated"] == pytest.approx(1.0)

    # Derived rate is always computed from numerator/denominator.
    assert out["rollout/parse_truncated_rate"] == pytest.approx(1.0 / 3.0)


def test_stage2_pending_log_finalize_uses_segment_weight_when_provided() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 10.0,
            "stage2/_log_weight": 1.0,
            "stage2/raw_rollouts": 1.0,
        }
    )
    pending.add(
        {
            "loss/B_coord/bbox_smoothl1": 20.0,
            "stage2/_log_weight": 3.0,
            "stage2/raw_rollouts": 2.0,
        }
    )

    out = pending.finalize()

    assert out["loss/B_coord/bbox_smoothl1"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert out["stage2/raw_rollouts"] == pytest.approx(3.0)
    assert "stage2/_log_weight_total" not in out


def test_stage2_pending_log_counter_suffixes_sum_not_weighted() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/A1_text/struct_ce": 1.0,
            "rollout/fp_total": 2.0,
            "rollout/fn_total": 1.0,
            "rollout/matched_maskiou_count": 3.0,
            "stage2/_log_weight": 1.0,
        }
    )
    pending.add(
        {
            "loss/A1_text/struct_ce": 3.0,
            "rollout/fp_total": 5.0,
            "rollout/fn_total": 4.0,
            "rollout/matched_maskiou_count": 7.0,
            "stage2/_log_weight": 3.0,
        }
    )

    out = pending.finalize()

    # Mean-like loss keys are weighted by stage2/_log_weight.
    assert out["loss/A1_text/struct_ce"] == pytest.approx(
        (1.0 * 1.0 + 3.0 * 3.0) / 4.0
    )

    # Counter-like keys with suffixes are always summed.
    assert out["rollout/fp_total"] == pytest.approx(7.0)
    assert out["rollout/fn_total"] == pytest.approx(5.0)
    assert out["rollout/matched_maskiou_count"] == pytest.approx(10.0)

    # Internal helper keys are removed from final payload.
    assert "stage2/_log_weight_total" not in out
    assert "rollout/_parse_truncated_num" not in out
    assert "rollout/_parse_truncated_den" not in out


def test_stage2_pending_log_emits_canonical_loss_prefix_only() -> None:
    pending = _PendingStage2Log()
    pending.add(
        {
            "loss/A1_text/struct_ce": 0.5,
            "loss/A1_text/desc_ce": 0.25,
            "loss/A2_coord/bbox_smoothl1": 0.25,
            "loss/A2_coord/coord_soft_ce": 0.125,
        }
    )

    out = pending.finalize()

    assert "loss/A1_text/struct_ce" in out
    assert "loss/A1_text/desc_ce" in out
    assert "loss/A2_coord/bbox_smoothl1" in out
    assert "loss/A2_coord/coord_soft_ce" in out
    assert "loss/token_ce_obj" not in out
    assert "loss/bbox_geo_obj" not in out
    assert "loss/coord_reg_obj" not in out
