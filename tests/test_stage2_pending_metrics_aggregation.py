from __future__ import annotations

import pytest

from src.trainers.stage2_ab_training import _PendingStage2Log


def test_stage2_pending_log_finalize_averages_losses_and_sums_counters() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/bbox_smoothl1": 1.0,
            "loss/coord_reg": 2.0,
            "stage2/channel_b": 1.0,
            "stage2/raw_rollouts": 1.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 1.0,
            "rollout/_parse_truncated_num": 1.0,
            "rollout/_parse_truncated_den": 2.0,
        }
    )
    pending.add(
        {
            "loss/bbox_smoothl1": 3.0,
            "loss/coord_reg": 4.0,
            "stage2/channel_b": 0.0,
            "stage2/raw_rollouts": 2.0,
            "rollout/seed_base": 10.0,
            "rollout/parse_truncated": 0.0,
            "rollout/_parse_truncated_num": 0.0,
            "rollout/_parse_truncated_den": 1.0,
        }
    )

    out = pending.finalize(drop_internal=False)

    # Averaged across micro-batches (n_micro=2).
    assert out["loss/bbox_smoothl1"] == pytest.approx(2.0)
    assert out["loss/coord_reg"] == pytest.approx(3.0)
    assert out["stage2/channel_b"] == pytest.approx(0.5)
    assert out["rollout/seed_base"] == pytest.approx(10.0)

    # Summed counters.
    assert out["stage2/raw_rollouts"] == pytest.approx(3.0)
    assert out["rollout/parse_truncated"] == pytest.approx(1.0)
    assert out["rollout/_parse_truncated_num"] == pytest.approx(1.0)
    assert out["rollout/_parse_truncated_den"] == pytest.approx(3.0)

    # Derived rate is always computed from numerator/denominator.
    assert out["rollout/parse_truncated_rate"] == pytest.approx(1.0 / 3.0)


def test_stage2_pending_log_finalize_uses_segment_weight_when_provided() -> None:
    pending = _PendingStage2Log()

    pending.add(
        {
            "loss/bbox_smoothl1": 10.0,
            "stage2/_log_weight": 1.0,
            "stage2/raw_rollouts": 1.0,
        }
    )
    pending.add(
        {
            "loss/bbox_smoothl1": 20.0,
            "stage2/_log_weight": 3.0,
            "stage2/raw_rollouts": 2.0,
        }
    )

    out = pending.finalize()

    assert out["loss/bbox_smoothl1"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert out["stage2/raw_rollouts"] == pytest.approx(3.0)
    assert "stage2/_log_weight_total" not in out

